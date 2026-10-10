"""Inference smoke test for the two added features (no sim env, no trained ckpt):
  1. parallel action-expert sampling (batched draft generation)
  2. verification reusing the sampling-stage KV cache (with safe fallback)

Random-init action expert + the (pretrained) Qwen2.5-VL-Action backbone.
Exercises ReasoningVLAModel.get_action(use_spec=True) end-to-end on fake inputs.
"""
import os, sys, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

sys.path.insert(0, ".")
from model.reasoning_vla_model import ReasoningVLAModel
from model.flow_matching_action_head import FlowmatchingActionHeadConfig

ROOT = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm-hlsc/hadoop-mlm/zhaochen"
VLM = f"{ROOT}/mr-ds/starvla_data/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action"
FAST = f"{ROOT}/starVLA/fast"

PAD = 64          # max_pad_state_and_action_length
HORIZON = 32
ACTION_DIM = 7
N_DRAFTS = 16     # action_sampling_times
N_STEPS = 4       # number of inference steps to run


def main():
    dev = "cuda"
    cfg = FlowmatchingActionHeadConfig(action_dim=PAD, state_dim=PAD, action_horizon=HORIZON)
    print("[smoke] building ReasoningVLAModel (random action expert) ...", flush=True)
    model = ReasoningVLAModel(
        vlm_name_or_path=VLM, action_expert_name_or_path=None, action_expert_config=cfg,
        tune_llm=False, tune_visual=False, tune_projector=False, tune_dit=False,
    ).to(dev).to(torch.bfloat16)
    model.eval()
    fast_tokenizer = AutoProcessor.from_pretrained(FAST, trust_remote_code=True)

    # --- fake observation -> VLM inputs via the real processor ---
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    msg = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "Task: pick up the red block."},
    ]}]
    text = model.backbone.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(msg)
    vl = model.backbone.processor(text=[text], images=imgs, return_tensors="pt")

    batch = {
        "input_ids": vl["input_ids"], "attention_mask": vl["attention_mask"],
        "pixel_values": vl["pixel_values"], "image_grid_thw": vl["image_grid_thw"],
        "observation.state": torch.randn(1, 1, PAD),
        "state_mask": torch.ones(1, 1, PAD, dtype=torch.long),
        "embodiment_id": torch.tensor([0], dtype=torch.long),
        "infer_action_mask": torch.ones(1, HORIZON, PAD, dtype=torch.long),
    }

    # ---------- baseline: NORMAL inference (use_spec=False) ----------
    # No CoT generation, no verification, single action-expert draft.
    print("[smoke] === baseline: get_action(use_spec=False) (normal 1-sample inference) ===", flush=True)
    base_t = []
    for step in range(N_STEPS):
        batch["observation.state"] = torch.randn(1, 1, PAD)
        torch.cuda.synchronize(); t0 = time.time()
        out = model.get_action(
            batch, num_denoised_steps=2, n_action_steps=HORIZON, use_spec=False,
            action_sampling_times=N_DRAFTS, action_dim=ACTION_DIM, action_tokenizer=fast_tokenizer,
        )
        torch.cuda.synchronize(); dt = time.time() - t0
        base_t.append(dt)
        print(f"[smoke] baseline step {step}: {dt:.2f}s | action_pred={tuple(out['action_pred'].shape)} "
              f"| modify_action={out['modify_action']}", flush=True)
    print(f"[smoke] baseline per-step: {[round(t,2) for t in base_t]} | mean(excl warmup)={np.mean(base_t[1:]):.2f}s", flush=True)

    print(f"[smoke] running {N_STEPS} inference steps, get_action(use_spec=True, action_sampling_times={N_DRAFTS}) ...", flush=True)
    timings = []
    for step in range(N_STEPS):
        # fresh random state each step (re-uses the same image/prompt for simplicity)
        batch["observation.state"] = torch.randn(1, 1, PAD)
        torch.cuda.synchronize()
        t0 = time.time()
        out = model.get_action(
            batch, num_denoised_steps=2, n_action_steps=HORIZON, use_spec=True,
            action_sampling_times=N_DRAFTS, action_dim=ACTION_DIM, action_tokenizer=fast_tokenizer,
        )
        torch.cuda.synchronize()
        dt = time.time() - t0
        timings.append(dt)

        cands = out["candicate_list"]
        pred = out["action_pred"]
        best = out["modify_action"]
        print(f"[smoke] step {step}: {dt:.2f}s | #candidates={len(cands)} "
              f"| cand={tuple(cands[0].shape)} | best={tuple(best.shape) if best is not None else None}",
              flush=True)
        assert len(cands) == N_DRAFTS + 1, "parallel sampling produced wrong #candidates"
        assert tuple(cands[0].shape) == (HORIZON, ACTION_DIM)
        assert best is not None and tuple(best.shape) == (HORIZON, ACTION_DIM)
        assert torch.isfinite(best.float()).all()

    print(f"[smoke] per-step times: {[round(t,2) for t in timings]}", flush=True)
    print(f"[smoke] mean (excl. 1st warmup) = {np.mean(timings[1:]):.2f}s", flush=True)
    print("[smoke] PARALLEL-SAMPLING + KV-REUSE-VERIFY OK (16 candidates, multi-step)", flush=True)
    print("[smoke] ALL GOOD", flush=True)


if __name__ == "__main__":
    main()
