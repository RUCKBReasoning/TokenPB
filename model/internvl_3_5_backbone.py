import torch
import time

from torch import nn
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from torch.nn import CrossEntropyLoss
import random
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class InternVL3_5_Backbone(nn.Module):
    def __init__(
        self, model_name: str, tune_llm: bool, tune_visual: bool, attn_implementation: str
    ):
        super().__init__()

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation
        )
        if tune_llm or tune_visual:
            self.model.gradient_checkpointing_enable()
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        
        self.set_trainable_parameters()

    def set_trainable_parameters(self):
        if not self.tune_llm:
            self.model.language_model.requires_grad_(False)
        if not self.tune_visual:
            self.model.vision_tower.requires_grad_(False)

    def set_frozen_modules_to_eval_mode(self):
        if self.training:
            if not self.tune_llm:
                self.model.language_model.eval()
            if not self.tune_visual:
                self.model.vision_tower.eval()

    def prepare_inputs(self, batch: dict):
        vl_inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "pixel_values": batch["pixel_values"]
        }

        if "labels" in batch:
            vl_inputs["labels"] = batch["labels"]
        
        return BatchFeature(data=vl_inputs).to(self.device)

    def generate(self, vl_inputs: BatchFeature, add_cot = False) -> BatchFeature:
        # TODO: we only support batch size = 1 during the inference
        if vl_inputs["input_ids"].shape[0] > 1:
            raise ValueError("we only support batch size = 1 during the inference, current batch size is: ", vl_inputs["input_ids"].shape[0])

        st = time.time()
        generated_seq = None
        seq_length = vl_inputs["input_ids"].shape[1]
        if add_cot:
            with torch.no_grad():
                outputs = self.model.generate(
                    **vl_inputs,
                    return_dict_in_generate=True,
                    max_new_tokens=1024,
                )
            seq_ids = outputs.sequences 
            generated_ids_trimmed = [
                seq_id[len(in_id) :] for in_id, seq_id in zip(vl_inputs["input_ids"], seq_ids)
            ]
            generated_seq = self.processor.batch_decode(generated_ids_trimmed)[0] # bs = 1
            print(generated_seq)
            new_attention_mask = (seq_ids != self.processor.tokenizer.pad_token_id).long() # (batch size, input+output len)
            vl_inputs["attention_mask"] = new_attention_mask
            # # update input_ids to be the completed sequences
            vl_inputs["input_ids"] = seq_ids
        # obtain the VLM's last layer hidden states
        vlm_outputs = self.forward(vl_inputs)
        # print("sampling score: ", self.get_score_by_logits(seq_ids[...,seq_length:], vlm_outputs['logits'][...,seq_length:,:]))
        print(f"vlm.forward() spends {time.time() - st} seconds")

        return vlm_outputs, generated_seq
    
    def find_min_index(self, tensor_list):
        # 将所有张量值转换为CPU上的Python数值
        values = [t.item() for t in tensor_list]
        # 找到最小值的索引
        min_index = values.index(min(values))
        return min_index
    
    def get_score_by_logits(self, action_label_list, logits):
        with torch.no_grad():
            loss_fnc = CrossEntropyLoss()
            action_score_list = []
            # print("padding: ", loss_fnc(logits[0], torch.tensor(action_label_list[0]).to(logits.device)))
            for idx, action_label in enumerate(action_label_list): # action_label_list is padding for DiT embedding
                action_label = action_label.to(logits.device)
                mask = (action_label == 151643)
                indices = torch.nonzero(mask, as_tuple=True)[0]
                if indices.numel() > 0:  # 检查是否找到了至少一个元素
                    first_index = indices[0].item() 
                    action_label = action_label[:first_index]
                    this_logits = logits[idx][:first_index,:]
                else:
                    this_logits = logits[idx]
                print(action_label, self.processor.tokenizer.decode(action_label), this_logits.shape, action_label.shape)
                action_loss = loss_fnc(this_logits[:-1, :], action_label[1:])
                action_score_list.append(action_loss / action_label.shape[0])
            index = self.find_min_index(action_score_list)
            print("action_score_list: ", action_score_list)
        return action_score_list, index
    
    def _trans_action_to_token(self, action: torch.Tensor, n_bins: int = 64):
        bins = np.linspace(-1, 1, n_bins)
        action = np.clip(action, a_min=-1., a_max=1.)
        discretized_action = np.digitize(action, bins)
        # print(action, discretized_action)
        return discretized_action
    
    def get_modify_action(self, vl_inputs, action_batch_list, real_action, action_tokenizer, desc_type = "bpe", backbone_outputs = None):
        # `backbone_outputs` accepted for API parity with the Qwen backbone (which
        # uses it for prefix KV-cache reuse); InternVL keeps the full-recompute path.
        import copy
        if not action_batch_list:
            return None, None
        st = time.time()
        action_list = copy.deepcopy(action_batch_list)
        im_start = torch.tensor(self.processor.tokenizer("<action>\n")["input_ids"])
        im_end = torch.tensor(self.processor.tokenizer("\n</action><|im_end|>")["input_ids"])
        # action_batch_list = [torch.cat((input_ids, im_start, action_sample, real_action, im_end), dim=0) for action_sample in action_batch_list]
        # action_batch_list = [action_tokenizer(torch.cat((action_sample, real_action), dim=0).cpu().long())[0] for action_sample in action_batch_list]
        # action_batch_list = [action_tokenizer(torch.tensor(action_sample).cpu().long())[0] for action_sample in action_batch_list]
        if desc_type == "bpe":
            action_batch_list = [self.processor.tokenizer(str(action_tokenizer(torch.tensor(action_sample.cpu().float()))))["input_ids"] for action_sample in action_batch_list]
        elif desc_type == "bins":
            action_batch_list = [torch.tensor(self._trans_action_to_token(action_sample.cpu().float())).view(-1) for action_sample in action_batch_list]
        elif desc_type == "qwen":
            action_batch_list = [torch.tensor(action_tokenizer(action_sample.cpu().float())[0]) for action_sample in action_batch_list]
            
            
            
        # print(action_batch_list)
        vl_len = len(vl_inputs['input_ids'][0].cpu())
        vl_inputs['input_ids'] = vl_inputs['input_ids'].repeat(len(action_batch_list), 1)
        new_input_ids_list = []
        
        for idx, action_sample in enumerate(action_batch_list):
            new_input_ids_list.append(torch.cat((vl_inputs['input_ids'][idx].cpu(), im_start, torch.tensor(action_sample), im_end), dim=0))
            
        padded_input_ids = pad_sequence(new_input_ids_list, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)
        vl_inputs['input_ids'] = padded_input_ids.to(vl_inputs['input_ids'].device)
        
        new_attention_mask = (vl_inputs['input_ids'] != self.processor.tokenizer.pad_token_id).long() # (batch size, input+output len)
        vl_inputs["attention_mask"] = new_attention_mask
        vl_inputs["input_ids"] = vl_inputs['input_ids']
        vl_inputs["pixel_values"] = vl_inputs["pixel_values"].repeat(len(action_batch_list), 1, 1, 1)
        print(f"spec prepare spends {time.time()-st} seconds.")
        
        backbone_logits = self.forward(vl_inputs)['logits'].cpu()
        print(f"spec forward spends {time.time()-st} seconds.")
        
        print(vl_inputs['input_ids'].shape, backbone_logits.shape)
        action_rank_score, index = self.get_score_by_logits(vl_inputs['input_ids'][...,vl_len:], backbone_logits[:,vl_len:, ...])
        best_action = action_list[index]
        print(f"vla.spec() spends {time.time()-st} seconds.")
        return action_rank_score, best_action

    def extract_action_expert_cross_attn_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: LongTensor, shape (bs, L)
        Returns: BoolTensor, shape (bs, L)

        Rule: For each row, set True for positions before and including the last occurrence (guaranteed unique) of the token ids [151644, 77091, 198], 
                and False for positions after.
        """
        # 151644 is '<|im_start|>', 77091 is 'assistant'; 198 is '\n'. 
        # They are adjacent tokens in the generation prompt
        a, b, c = 151644, 77091, 198
        bs, L = input_ids.shape
        if L < 3:
            return torch.ones((bs, L), dtype=torch.bool, device=input_ids.device)

        # Ternary neighbor matching yields a Boolean tensor of shape (bs, L-2), 
        # with True indicating that the position is the start of the ternary sequence
        matches3 = (input_ids[:, :-2] == a) & (input_ids[:, 1:-1] == b) & (input_ids[:, 2:] == c)

        # Each sequence appears exactly once, use argmax to take the starting position
        start_idx = matches3.to(torch.int64).argmax(dim=1)  # (bs,)
        end_idx = start_idx + 2  # position of the last token of the ternary sequence

        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)  # (1, L)
        mask = pos <= end_idx.unsqueeze(1)  # (bs, L), bool
        return mask

    def forward(self, vl_inputs: BatchFeature) -> BatchFeature:
        # set frozen module to eval
        self.set_frozen_modules_to_eval_mode()

        use_cache = False if self.training else True
        model_outputs = self.model(**vl_inputs, return_dict=True, 
                                   output_hidden_states=True, use_cache=use_cache)
        
        # return the embeddings of all tokens in the last layer (bs, seq_len, hidden_size)
        embeddings = model_outputs["hidden_states"][-1] # e.g., (4, 4096, 2048)
        if random.random() < 0.01:
            # print("vlm forward embeddings: ", embeddings)
            print("image_hidden_states: ", model_outputs['image_hidden_states'])
            

        # let the action expert only attend to the input (i.e., "image+text prompt" part) of the VLM,
        # including the generation prompt '<|im_start|>assistant\n'. Thus, during inference, we should set `add_generation_prompt` to True.
        action_expert_cross_attn_mask = self.extract_action_expert_cross_attn_mask(vl_inputs["input_ids"])
        
        # if torch.cuda.current_device() == 0:
        #     for token_id, mask in zip(vl_inputs["input_ids"][0], action_expert_cross_attn_mask[0]):
        #         print(token_id.item(), "->", repr(self.processor.tokenizer.decode(token_id)), "->", mask.item())
        #     print("*"*50)

        return BatchFeature(
            data={
                "backbone_embeddings": embeddings,
                "action_expert_cross_attn_mask": action_expert_cross_attn_mask,
                "logits": model_outputs['logits'],
                "vlm_loss": model_outputs["loss"] if "labels" in vl_inputs else None
            }
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device
