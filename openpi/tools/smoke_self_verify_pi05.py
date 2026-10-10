"""Smoke test: self-verification on a random-init pi0.5 (PyTorch port).

Uses the real gemma_2b / gemma_300m variants (consistent dims) with RANDOM
weights — no base checkpoint needed. Validates that, with self_verify on:
  * forward() returns (mse, aux_fast_loss) and aux is finite,
  * sample_actions() runs draft-and-verify and returns a single [1,H,D] action.
Random weights => meaningless values, but shapes/flow are exercised end-to-end.

Run (needs proxy for the small tokenizer downloads):
  python tools/smoke_self_verify_pi05.py
"""
import numpy as np
import torch

from openpi.models import pi0_config
from openpi.models.model import Observation
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


def main():
    device = "cuda"
    cfg = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pytorch_compile_mode=None,
        self_verify=True,
        self_verify_num_candidates=4,
    )
    print(f"[smoke] building pi0.5 (self_verify={cfg.self_verify}) random init ...")
    model = PI0Pytorch(cfg).to(device)
    model.eval()
    print(f"[smoke] _sv_enabled={model._sv_enabled} num_candidates={model._sv_num_candidates}")
    assert model._sv_enabled

    B = 2
    H, D, L = cfg.action_horizon, cfg.action_dim, cfg.max_token_len
    cams = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]

    def make_obs(bs):
        img = {c: torch.rand(bs, 224, 224, 3, device=device) * 2 - 1 for c in cams}
        imask = {c: torch.ones(bs, dtype=torch.bool, device=device) for c in cams}
        # random-but-valid prompt token ids + mask (left part valid)
        tok = torch.randint(4, 257000, (bs, L), dtype=torch.long, device=device)
        tmask = torch.zeros(bs, L, dtype=torch.bool, device=device)
        tmask[:, :16] = True  # first 16 tokens are the "prompt"
        state = torch.rand(bs, D, device=device) * 2 - 1
        return Observation.from_dict(
            {
                "image": img,
                "image_mask": imask,
                "state": state,
                "tokenized_prompt": tok,
                "tokenized_prompt_mask": tmask,
            }
        )

    # --- training forward ---
    obs = make_obs(B)
    actions = torch.rand(B, H, D, device=device) * 2 - 1
    model.train()
    out = model(obs, actions)
    assert isinstance(out, tuple) and len(out) == 2, f"expected (mse, aux), got {type(out)}"
    mse, aux = out
    total = mse.mean() + aux
    print(f"[smoke] forward: mse.mean()={mse.mean().item():.4f} fast_loss(weighted)={aux.item():.4f} total={total.item():.4f}")
    assert torch.isfinite(aux), "aux fast loss not finite"
    total.backward()
    g = [p for n, p in model.named_parameters() if "paligemma" in n and p.grad is not None]
    print(f"[smoke] #PaliGemma params with grad: {len(g)}")
    assert len(g) > 0, "no grad reached the VLM — verifier not training"
    model.zero_grad(set_to_none=True)
    print("[smoke] TRAIN OK")

    # --- inference draft-and-verify (B=1) ---
    model.eval()
    obs1 = make_obs(1)
    with torch.no_grad():
        act = model.sample_actions(device, obs1, num_steps=4)
    print(f"[smoke] sample_actions (verify) shape: {tuple(act.shape)}")
    assert tuple(act.shape) == (1, H, D), f"expected (1,{H},{D}), got {tuple(act.shape)}"
    assert torch.isfinite(act).all()
    print("[smoke] INFER (draft-and-verify) OK")
    print("[smoke] ALL GOOD")


if __name__ == "__main__":
    main()
