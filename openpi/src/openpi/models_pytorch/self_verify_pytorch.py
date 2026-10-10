# Self-Verification (Action Draft & Verify) for the PyTorch pi0.5 port.
#
# Implements the optional self-verification mechanism from "Action Draft and
# Verify" (arXiv:2603.18091) for openpi's PyTorch PI0Pytorch model.
#
# pi0.5 has a flow-matching action expert (the "draft" generator) and a
# PaliGemma VLM, but NO autoregressive action head. This mixin adds one:
#   * Training: an auxiliary cross-entropy loss where the PaliGemma LLM predicts
#     the FAST tokens of the ground-truth action chunk (appended to the prompt).
#   * Inference: the flow head samples N candidate action chunks ("drafts"); each
#     is FAST-tokenized, appended to the prompt, and scored by the LLM token
#     likelihood; the most-likely candidate is returned ("verify").
#
# The verifier sequence is built by appending fast(action) PaliGemma tokens +
# "|"+EOS to the model's existing ``tokenized_prompt`` (format-agnostic; works
# regardless of discrete_state_input). Logits are computed from the tied input
# embedding, mirroring openpi's Pi0FAST.
#
# Fully gated by ``config.self_verify``; when off, none of this runs.

import logging
import math

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

logger = logging.getLogger("openpi")


class SelfVerifyMixin:
    """Mixin adding optional FAST-token self-verification to ``PI0Pytorch``.

    Host model must provide: ``paligemma_with_expert`` (with ``embed_image``,
    ``embed_language_tokens``, ``forward``), ``config`` (with ``action_horizon``,
    ``action_dim``, ``self_verify*`` fields), ``make_att_2d_masks`` /
    ``_prepare_attention_masks_4d`` helpers, and ``_preprocess_observation``.
    """

    # ------------------------------------------------------------------ setup
    def _sv_init(self):
        """Build the FAST tokenizer (lazy) and cache constants. Call from __init__."""
        self._sv_enabled = bool(getattr(self.config, "self_verify", False))
        self._sv_loss_weight = float(getattr(self.config, "self_verify_loss_weight", 1.0))
        self._sv_num_candidates = int(getattr(self.config, "self_verify_num_candidates", 8))
        self._sv_fast_path = str(getattr(self.config, "fast_tokenizer_path", "physical-intelligence/fast"))
        self._sv_tokenizer = None
        if not self._sv_enabled:
            return

        from openpi.models.tokenizer import FASTTokenizer

        # max_len here only bounds FASTTokenizer.tokenize(); we use its internals.
        self._sv_tokenizer = FASTTokenizer(max_len=512, fast_tokenizer_path=self._sv_fast_path)
        pg = self._sv_tokenizer._paligemma_tokenizer  # noqa: SLF001
        # Fixed suffix tokens: "|" followed by EOS (closes the FAST action span).
        self._sv_suffix_ids = list(pg.encode("|", add_eos=True))
        logger.info(
            f"[self_verify] ENABLED | num_candidates={self._sv_num_candidates} "
            f"| loss_weight={self._sv_loss_weight} | fast_tokenizer={self._sv_fast_path}"
        )

    # ----------------------------------------------------------- tokenization
    def _sv_action_to_pg_ids(self, action_np: np.ndarray) -> list[int]:
        """Continuous action chunk [H, D] -> FAST-in-PaliGemma token ids + suffix."""
        tok = self._sv_tokenizer
        fast_ids = tok._fast_tokenizer(action_np[None])[0]  # noqa: SLF001
        action_pg = tok._act_tokens_to_paligemma_tokens(fast_ids).tolist()  # noqa: SLF001
        return [int(x) for x in action_pg] + self._sv_suffix_ids

    def _sv_build_batch(self, tokenized_prompt, prompt_mask, actions_np_list, device):
        """Build padded (tokens, pad_mask, ar_mask, loss_mask) for a batch.

        Each row = [valid prompt tokens] + [fast(action) + "|"+EOS].
        ar_mask: 0 on prompt (bidirectional), 1 on the appended action span (causal).
        loss_mask: True only on the appended action span.
        """
        prompt_ids = tokenized_prompt.detach().cpu().numpy()
        masks = prompt_mask.detach().cpu().numpy().astype(bool)
        rows_tokens, rows_ar, rows_loss = [], [], []
        for b, action_np in enumerate(actions_np_list):
            valid = [int(t) for t in prompt_ids[b][masks[b]]]
            action_part = self._sv_action_to_pg_ids(action_np)
            toks = valid + action_part
            ar = [0] * len(valid) + [1] * len(action_part)
            loss = [False] * len(valid) + [True] * len(action_part)
            rows_tokens.append(toks)
            rows_ar.append(ar)
            rows_loss.append(loss)

        max_len = max(len(t) for t in rows_tokens)
        n = len(rows_tokens)
        tokens = torch.zeros(n, max_len, dtype=torch.long, device=device)
        pad = torch.zeros(n, max_len, dtype=torch.bool, device=device)
        ar = torch.zeros(n, max_len, dtype=torch.long, device=device)
        loss = torch.zeros(n, max_len, dtype=torch.bool, device=device)
        for i, (t, a, l) in enumerate(zip(rows_tokens, rows_ar, rows_loss, strict=True)):
            ln = len(t)
            tokens[i, :ln] = torch.tensor(t, dtype=torch.long, device=device)
            pad[i, :ln] = True
            ar[i, :ln] = torch.tensor(a, dtype=torch.long, device=device)
            loss[i, :ln] = torch.tensor(l, dtype=torch.bool, device=device)
        return tokens, pad, ar, loss

    # --------------------------------------------------------------- LM logits
    def _sv_prefix_logits(self, images, img_masks, ext_tokens, ext_pad, ext_ar):
        """Embed images + extended tokens, run the PaliGemma LLM, return logits.

        Returns logits over the language-token span only ``[B, L_lang, vocab]``
        (image positions are not decoded, to save the large vocab matmul).
        """
        pwe = self.paligemma_with_expert
        embs, pads, ars = [], [], []
        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = pwe.embed_image(img)
            bsize, n_img = img_emb.shape[:2]
            embs.append(img_emb)
            pads.append(img_mask[:, None].expand(bsize, n_img))
            ars.append(torch.zeros(bsize, n_img, dtype=torch.long, device=img_emb.device))

        n_img_total = sum(e.shape[1] for e in embs)
        lang_emb = pwe.embed_language_tokens(ext_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pads.append(ext_pad)
        ars.append(ext_ar)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pads, dim=1)
        att_masks = torch.cat(ars, dim=1).to(torch.bool)

        if embs.dtype != lang_emb.dtype:
            embs = embs.to(lang_emb.dtype)
        # Match the LLM parameter dtype (bf16 unless float32 precision).
        param_dtype = pwe.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        if param_dtype == torch.bfloat16:
            embs = embs.to(torch.bfloat16)

        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        att_2d = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_4d = self._prepare_attention_masks_4d(att_2d)
        pwe.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        (prefix_out, _), _ = pwe.forward(
            attention_mask=att_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[embs, None],
            use_cache=False,
        )
        lang_hidden = prefix_out[:, n_img_total:].to(torch.float32)
        embed_w = pwe.paligemma.language_model.embed_tokens.weight.to(torch.float32)
        logits = F.linear(lang_hidden, embed_w)  # [B, L_lang, vocab]
        return logits

    @staticmethod
    def _sv_token_ce(logits, ext_tokens, loss_mask):
        """Per-sample mean cross-entropy over the masked (action) token positions.

        Lower = the action sequence is more likely under the VLM.
        """
        pred = logits[:, :-1, :]
        target = ext_tokens[:, 1:]
        mask = loss_mask[:, 1:].to(torch.float32)
        logp = F.log_softmax(pred, dim=-1)
        tok_lp = torch.gather(logp, -1, target[..., None].clamp(min=0))[..., 0]
        return -(tok_lp * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    # ---------------------------------------------------------------- training
    def _sv_compute_loss(self, images, img_masks, tokenized_prompt, prompt_mask, actions):
        """Auxiliary FAST-token AR loss for the GT actions (scalar)."""
        actions_np = actions.detach().to(torch.float32).cpu().numpy()
        actions_list = [actions_np[b] for b in range(actions_np.shape[0])]
        ext_tokens, ext_pad, ext_ar, ext_loss = self._sv_build_batch(
            tokenized_prompt, prompt_mask, actions_list, actions.device
        )
        logits = self._sv_prefix_logits(images, img_masks, ext_tokens, ext_pad, ext_ar)
        per_sample = self._sv_token_ce(logits, ext_tokens, ext_loss)
        return per_sample.mean()

    # --------------------------------------------------------------- inference
    @torch.no_grad()
    def _sv_select(self, images, img_masks, tokenized_prompt, prompt_mask, candidate_actions):
        """Score candidate action chunks; return (best_index, scores[N]).

        ``candidate_actions``: tensor [N, H, D] (drafts for a single observation).
        ``images``/``img_masks``: single-observation tensors (will be tiled to N).
        """
        num = candidate_actions.shape[0]
        cand_np = candidate_actions.detach().to(torch.float32).cpu().numpy()
        cand_list = [cand_np[i] for i in range(num)]
        # tile the single observation to N rows
        img_n = [img.expand(num, *img.shape[1:]) for img in images]
        mask_n = [m.expand(num, *m.shape[1:]) for m in img_masks]
        prompt_n = tokenized_prompt.expand(num, *tokenized_prompt.shape[1:])
        pmask_n = prompt_mask.expand(num, *prompt_mask.shape[1:])
        ext_tokens, ext_pad, ext_ar, ext_loss = self._sv_build_batch(
            prompt_n, pmask_n, cand_list, candidate_actions.device
        )
        logits = self._sv_prefix_logits(img_n, mask_n, ext_tokens, ext_pad, ext_ar)
        scores = self._sv_token_ce(logits, ext_tokens, ext_loss)
        best_idx = int(torch.argmin(scores).item())
        return best_idx, scores
