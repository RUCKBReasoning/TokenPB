from transformers.feature_extraction_utils import BatchFeature
from .qwen2_5_vl_backbone import Qwen2_5VLBackbone
from .internvl_3_5_backbone import InternVL3_5_Backbone
from torch import nn
from typing import Tuple
from .flow_matching_action_head import FlowmatchingActionHead, FlowmatchingActionHeadConfig

import torch
import json
import os
import copy
from safetensors.torch import save_file, load_file

class ReasoningVLAModel(nn.Module):
    def __init__(
            self,
            vlm_name_or_path: str,
            action_expert_name_or_path: str,
            action_expert_config: FlowmatchingActionHeadConfig,
            tune_llm=True,
            tune_visual=True,
            tune_projector=True,
            tune_dit=True,
            attn_implementation="flash_attention_2",
            detach_vlm_outputs_for_action_expert=False,
            backbone:str = None
        ):
        super().__init__()

        self.action_expert_config = action_expert_config
        if "Qwen" in vlm_name_or_path or backbone == "qwen2.5vl":
            self.backbone = Qwen2_5VLBackbone(vlm_name_or_path, tune_llm, tune_visual, attn_implementation)
        elif "InternVL" in vlm_name_or_path or backbone == "internvl3.5":
            self.backbone = InternVL3_5_Backbone(vlm_name_or_path, tune_llm, tune_visual, attn_implementation)
        else:
            raise NotImplementedError(f"backbone {vlm_name_or_path} not implemented yet!")
        self.action_expert_config.vlm_output_embedding_dim = 2048  # fixed for InternVL3.5
        self.action_expert = FlowmatchingActionHead(self.action_expert_config, tune_projector, tune_dit)
        self.detach_vlm_outputs_for_action_expert = detach_vlm_outputs_for_action_expert

        if action_expert_name_or_path:
            print(f"load pre-trained weights from {action_expert_name_or_path} to initialize the action expert")
            self.action_expert.load_state_dict(
                load_file(os.path.join(action_expert_name_or_path, "action_expert.safetensors"))
            )

    def prepare_vla_input(self, batch_inputs) -> Tuple[BatchFeature, BatchFeature]:
        backbone_inputs = self.backbone.prepare_inputs(batch_inputs)
        action_expert_inputs = self.action_expert.prepare_inputs(batch_inputs)
        # print(f"prepare a batched samples spends {time.time()-start_time}s")
        return backbone_inputs, action_expert_inputs

    @torch.inference_mode()
    def get_action(self, batch_inputs: BatchFeature,
                   num_denoised_steps: int = 10, n_action_steps: int = 32, use_spec = True, action_sampling_times = 15, action_batch_list = [], action_dim = 7, action_tokenizer = None) -> BatchFeature:
        
        with torch.no_grad():
            backbone_inputs = self.backbone.prepare_inputs(batch_inputs)
            # No CoT generation: only run the prefix forward (hidden states + KV
            # cache) regardless of use_spec. `use_spec` still controls the
            # draft-and-verify path below.
            backbone_outputs, best_action = self.backbone.generate(backbone_inputs, add_cot=False)

            action_expert_inputs = self.action_expert.prepare_inputs(batch_inputs)
            action_expert_inputs_backup = copy.deepcopy(action_expert_inputs)

            action_expert_outputs = self.action_expert.get_action(
                backbone_outputs, action_expert_inputs, num_denoised_steps
            )
            # print("action_expert_outputs: ",action_expert_outputs)
            # _, best_action = self.backbone.get_modify_action(backbone_inputs, action_batch_list, action_expert_outputs["action_pred"][0][n_action_steps//2 : n_action_steps, :action_dim], action_tokenizer)
            action_batch_list = []
            best_action = None
            if use_spec:
                assert n_action_steps % 4 == 0, f"n_action_steps ({n_action_steps}) must be divisible by 4!"
                action_batch_list.append(action_expert_outputs["action_pred"][0][:32, :action_dim])
                # Parallel draft sampling: sample all remaining candidates in ONE
                # batched action-expert call instead of the serial Python loop.
                # The flow-matching head already runs over the batch dim, so
                # repeating the (shared) VLM conditioning + state inputs to
                # `action_sampling_times` rows produces that many independent
                # denoised trajectories (independent initial noise per row).
                if action_sampling_times > 0:
                    bo_rep = self._repeat_backbone_outputs(backbone_outputs, action_sampling_times)
                    ae_rep = self._repeat_action_expert_inputs(action_expert_inputs_backup, action_sampling_times)
                    sampled = self.action_expert.get_action(
                        bo_rep, ae_rep, num_denoised_steps
                    )["action_pred"]  # (action_sampling_times, action_horizon, action_dim)
                    for i in range(action_sampling_times):
                        action_batch_list.append(sampled[i][:32, :action_dim])
                # Verify stage reuses the sampling-stage prefix KV cache (passed via
                # backbone_outputs) so the image+text prefix is not re-encoded.
                _, best_action = self.backbone.get_modify_action(
                    backbone_inputs, action_batch_list, None, action_tokenizer,
                    backbone_outputs=backbone_outputs,
                )

        return BatchFeature(data={"action_pred": action_expert_outputs["action_pred"], "candicate_list": action_batch_list, "modify_action": best_action})
        # return BatchFeature(data={"action_pred": action_batch_list[-1], "candicate_list": action_batch_list, "modify_action": best_action})
        
    def _repeat_backbone_outputs(self, backbone_outputs: BatchFeature, n: int) -> BatchFeature:
        """Repeat the VLM conditioning along the batch dim for parallel sampling.

        Only the tensors the action expert consumes are repeated.
        """
        return BatchFeature(data={
            "backbone_embeddings": backbone_outputs["backbone_embeddings"].repeat(n, 1, 1),
            "action_expert_cross_attn_mask": backbone_outputs["action_expert_cross_attn_mask"].repeat(n, 1),
        })

    def _repeat_action_expert_inputs(self, action_expert_inputs: BatchFeature, n: int) -> BatchFeature:
        """Repeat the action-expert inputs along the batch dim for parallel sampling."""
        data = {}
        for k, v in action_expert_inputs.items():
            if torch.is_tensor(v):
                data[k] = v.repeat(n, *([1] * (v.dim() - 1)))
            else:
                data[k] = v
        return BatchFeature(data=data).to(self.action_expert.device)

    def forward(self, batch_inputs) -> BatchFeature:
        backbone_inputs, action_expert_inputs = self.prepare_vla_input(batch_inputs)
        backbone_outputs = self.backbone(backbone_inputs)

        # if self.detach_vlm_outputs_for_action_expert:
        #     backbone_outputs["backbone_embeddings"] = backbone_outputs["backbone_embeddings"].detach()
        
        action_expert_outputs = self.action_expert(backbone_outputs, action_expert_inputs)

        # print("vlm_loss:", backbone_outputs["vlm_loss"])
        # print("action_expert_loss:", action_expert_outputs["action_expert_loss"])

        return BatchFeature(
            data={"loss": backbone_outputs["vlm_loss"] + action_expert_outputs["action_expert_loss"], "vlm_loss": backbone_outputs["vlm_loss"], "action_expert_loss": action_expert_outputs["action_expert_loss"]}
        )
    
    @property
    def device(self):
        return next(iter(self.parameters())).device

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        # save backbone model parameter and model config (Qwen2.5-VL)
        self.backbone.model.save_pretrained(save_directory)
        # save backnone's processor
        self.backbone.processor.save_pretrained(save_directory)
        # save action expert's model parameter
        save_file(self.action_expert.state_dict(), os.path.join(save_directory, "action_expert.safetensors"))
        # save action expert's config
        with open(os.path.join(save_directory, "action_expert_config.json"), "w") as json_file:
            json.dump(self.action_expert_config.to_dict(), json_file, indent=4)
    
    @classmethod
    def from_pretrained(cls, save_directory, tune_llm = False, tune_visual = False, tune_projector = False, 
                        tune_dit = False, attn_implementation="flash_attention_2",
                        detach_vlm_outputs_for_action_expert=False, backbone:str = None):
        # `save_directory` should contain complete VLA weights (a VLM and an action expert)
        action_expert_config_json = json.load(open(os.path.join(save_directory, "action_expert_config.json")))
        action_expert_config = FlowmatchingActionHeadConfig(**action_expert_config_json)

        return cls(
            vlm_name_or_path = save_directory,
            action_expert_name_or_path = save_directory,
            action_expert_config = action_expert_config,
            tune_llm = tune_llm,
            tune_visual = tune_visual,
            tune_projector = tune_projector,
            tune_dit = tune_dit,
            attn_implementation = attn_implementation,
            detach_vlm_outputs_for_action_expert = detach_vlm_outputs_for_action_expert,
            backbone = backbone
        )
