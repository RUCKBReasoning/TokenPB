set -e

# SFT with Qwen2.5-VL + random action expert

# date: 20251013
# batch size = 6*8*10 = 480
# epoch = 8
# lr = 2e-5
# dataset: [robotwin]
# NOTE: 
# 1. Action: delta of position (3-D) and orientation (3-D) + gripper state
# 2. Output: only-action (no discrete action tokens)
# 3. gradient stop between VLM and action expert
# 4. do not use discrete action
# 5. no episode start
# 6. window size = 1

# dump_bin_bigbin, grab_roller, hanging_mug

# use bpe token for spec decoding

CUDA_VISIBLE_DEVICES=3,4,5,6 accelerate launch --main_process_port 29501 --config_file ./accelerate_configs/accelerate_config_vla_1_node_develop.yaml \
    train_vla.py \
    --vlm_name_or_path "/workspace/zhaochen/.cache/OpenGVLab/InternVL3_5-2B-HF" \
    --FAST_tokenizer_path "/root/hf_models/physical-intelligence/fast" \
    --per_device_train_batch_size 4 \
    --seed 42 \
    --epochs 48 \
    --save_ckpt_interval 16 \
    --save_step_interval 3000 \
    --lr 8e-5 \
    --tensorboard_log_dir "/workspace/zhaochen/reasoningVLA-ruc/outputs/ckpts/train_logs/BaM-InernVL-2b-20251203" \
    --output_ckpt_dir "/workspace/zhaochen/reasoningVLA-ruc/outputs/ckpts/BaM-InernVL-2b-20251203" \
    --tune_visual \
    --tune_llm \
    --tune_projector \
    --tune_dit \
    --dataset_path_list "/workspace/dataset/sim_benchmarks/libero_v21" \
    --mask_prompt_loss \
    --add_discrete_action_tokens \
    --window_size 1 \
    --action_horizon 32 \
    --max_pad_state_and_action_length 64 \
    --desc_type "bpe"
