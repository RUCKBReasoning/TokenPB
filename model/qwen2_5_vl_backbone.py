import torch
import time

from torch import nn
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from transformers.feature_extraction_utils import BatchFeature
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random

class Qwen2_5VLBackbone(nn.Module):
    def __init__(
        self, model_name: str, tune_llm: bool, tune_visual: bool, attn_implementation: str
    ):
        super().__init__()

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )
        # FlashAttention-2 requires fp16/bf16 weights; cast the VLM to bf16
        # (consistent with the bf16 mixed-precision training setup).
        self.model = self.model.to(torch.bfloat16)
        if tune_llm or tune_visual:
            self.model.gradient_checkpointing_enable()
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_name)

        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        
        self.set_trainable_parameters()

    def set_trainable_parameters(self):
        if not self.tune_llm:
            self.model.model.requires_grad_(False)
        if not self.tune_visual:
            self.model.visual.requires_grad_(False)

    def set_frozen_modules_to_eval_mode(self):
        if self.training:
            if not self.tune_llm:
                self.model.model.eval()
            if not self.tune_visual:
                self.model.visual.eval()

    def prepare_inputs(self, batch: dict):
        vl_inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "pixel_values": batch["pixel_values"],
            "image_grid_thw": batch["image_grid_thw"]
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
                    do_sample=False,
                    # do_sample=True,
                    # temperature=0.8,
                    # num_return_sequences=1,
                    # top_p=0.95,
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
    
    def _expand_kv_cache(self, past_key_values, n: int):
        """Repeat a batch-size-1 KV cache along the batch dim to ``n``.

        Supports the legacy tuple format, the transformers 4.50 ``DynamicCache``
        (key_cache/value_cache lists), and the 4.57 ``DynamicCache`` (``.layers``
        with ``.keys``/``.values`` + a built-in ``batch_repeat_interleave``).
        """
        # Legacy tuple format: tuple(layers) of tuple(key, value)
        if isinstance(past_key_values, (tuple, list)):
            return tuple((layer[0].repeat(n, 1, 1, 1), layer[1].repeat(n, 1, 1, 1)) for layer in past_key_values)
        # transformers >= ~4.55: built-in batch repeat (in-place; the prefix cache
        # is single-use here, so mutation is fine).
        if hasattr(past_key_values, "batch_repeat_interleave"):
            past_key_values.batch_repeat_interleave(n)
            return past_key_values
        # transformers 4.50 DynamicCache
        if hasattr(past_key_values, "key_cache"):
            past_key_values.key_cache = [k.repeat(n, 1, 1, 1) for k in past_key_values.key_cache]
            past_key_values.value_cache = [v.repeat(n, 1, 1, 1) for v in past_key_values.value_cache]
            return past_key_values
        # transformers 4.57 DynamicCache (.layers with .keys/.values)
        if hasattr(past_key_values, "layers"):
            for layer in past_key_values.layers:
                layer.keys = layer.keys.repeat(n, 1, 1, 1)
                layer.values = layer.values.repeat(n, 1, 1, 1)
            return past_key_values
        raise TypeError(f"unknown KV cache type: {type(past_key_values)}")

    def get_modify_action(self, vl_inputs, action_batch_list, real_action, action_tokenizer, desc_type = "bpe", backbone_outputs = None):
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
            # Batched FAST encode (one call for all candidates) + batched BPE.
            # `str([fast_ids])` reproduces the original per-candidate `str(ft(2d))`
            # double-bracket format, so the scored representation is unchanged.
            acts = np.stack([np.asarray(a.detach().cpu().float()) for a in action_batch_list], axis=0)
            fast_batch = action_tokenizer(acts)
            strs = [str([fast_batch[i]]) for i in range(len(fast_batch))]
            action_batch_list = self.processor.tokenizer(strs)["input_ids"]
        elif desc_type == "bins":
            action_batch_list = [torch.tensor(self._trans_action_to_token(action_sample.cpu().float())).view(-1) for action_sample in action_batch_list]
        elif desc_type == "qwen":
            action_batch_list = [torch.tensor(action_tokenizer(action_sample.cpu().float())[0]) for action_sample in action_batch_list]

        n_cand = len(action_batch_list)
        pad_id = self.processor.tokenizer.pad_token_id

        # ── Fast path: reuse the sampling-stage prefix KV cache ───────────────
        # Only forward the appended action tokens against the cached image+text
        # prefix, instead of re-encoding the whole prefix for every candidate.
        prefix_cache = backbone_outputs.get("past_key_values", None) if backbone_outputs is not None else None
        if prefix_cache is not None:
            try:
                device = vl_inputs["input_ids"].device
                prefix_input_ids = backbone_outputs["prefix_input_ids"]
                prefix_attn = backbone_outputs["prefix_attention_mask"]
                image_grid_thw = backbone_outputs.get("prefix_image_grid_thw", vl_inputs.get("image_grid_thw", None))
                prefix_len = prefix_input_ids.shape[1]

                # Build the per-candidate suffix = <action>\n {tokens} \n</action><|im_end|>
                suffix_list = []
                for action_sample in action_batch_list:
                    suffix_list.append(torch.cat((im_start, torch.tensor(action_sample), im_end), dim=0))
                padded_suffix = pad_sequence(suffix_list, batch_first=True, padding_value=pad_id).to(device)
                suffix_len = padded_suffix.shape[1]

                # Attention mask covers the full (cached prefix + new suffix) sequence.
                suffix_mask = (padded_suffix != pad_id).long()
                prefix_mask = torch.ones((1, prefix_len), dtype=suffix_mask.dtype, device=device).repeat(n_cand, 1)
                full_mask = torch.cat([prefix_mask, suffix_mask], dim=1)

                # Continue the (m)rope positions right after the prefix. The suffix is
                # pure text, so all 3 Qwen2.5-VL mrope dims increment by 1 per token.
                get_rope_index = getattr(self.model, "get_rope_index", None) or self.model.model.get_rope_index
                prefix_pos, _ = get_rope_index(
                    prefix_input_ids, image_grid_thw=image_grid_thw, attention_mask=prefix_attn
                )  # (3, 1, prefix_len)
                start = int(prefix_pos.max().item()) + 1
                ar = torch.arange(suffix_len, device=device)
                suffix_pos = (start + ar).view(1, 1, suffix_len).expand(3, n_cand, suffix_len).contiguous()

                cache_n = self._expand_kv_cache(prefix_cache, n_cand)
                outputs = self.model(
                    input_ids=padded_suffix,
                    attention_mask=full_mask,
                    position_ids=suffix_pos,
                    past_key_values=cache_n,
                    use_cache=True,
                    return_dict=True,
                )
                # Score by perplexity ON GPU. Avoid moving the full-vocab logits
                # [N, L, V] to CPU (that transfer dominated the cost); instead
                # compute the per-candidate mean NLL of the action tokens here and
                # transfer only the N scalar scores. Logits already correspond to
                # the suffix positions (suffix-only forward).
                shift_logits = outputs["logits"][:, :-1, :].float()      # [N, L-1, V]
                shift_labels = padded_suffix[:, 1:]                       # [N, L-1]
                N, Lm1, V = shift_logits.shape
                ce = torch.nn.functional.cross_entropy(
                    shift_logits.reshape(-1, V), shift_labels.reshape(-1),
                    ignore_index=pad_id, reduction="none",
                ).view(N, Lm1)
                valid = (shift_labels != pad_id).float()
                nll = ce.sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)     # [N] perplexity proxy
                index = int(torch.argmin(nll).item())
                action_rank_score = nll.detach().cpu()
                best_action = action_list[index]
                print(f"spec forward (kv-cache reuse) spends {time.time()-st} seconds.")
                print(f"vla.spec() spends {time.time()-st} seconds.")
                return action_rank_score, best_action
            except Exception as e:
                print(f"[get_modify_action] KV-cache reuse failed ({e}); falling back to full recompute.")

        # ── Fallback path: full prefix+suffix recompute (original behaviour) ──
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
        vl_inputs["pixel_values"] = vl_inputs["pixel_values"].repeat(len(action_batch_list), 1)
        vl_inputs["image_grid_thw"] = vl_inputs["image_grid_thw"].repeat(len(action_batch_list), 1)
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
            print("vlm forward embeddings: ", embeddings)
            print("vlm forward embeddings shape: ", embeddings.shape)

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
                "vlm_loss": model_outputs["loss"] if "labels" in vl_inputs else None,
                # Prefix KV cache + the inputs needed to reuse it during verification
                # (Action Draft & Verify: verify stage reuses the sampling-stage cache).
                "past_key_values": model_outputs.get("past_key_values", None) if use_cache else None,
                "prefix_input_ids": vl_inputs["input_ids"],
                "prefix_attention_mask": vl_inputs["attention_mask"],
                "prefix_image_grid_thw": vl_inputs.get("image_grid_thw", None),
            }
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device
