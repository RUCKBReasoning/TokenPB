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

# stack_blocks_three-aloha-agilex_clean_50-50, stack_blocks_two-aloha-agilex_clean_50-50, stack_bowls_three-aloha-agilex_clean_50-50, stack_bowls_two-aloha-agilex_clean_50-50

# use bpe token for spec decoding

accelerate launch --main_process_port 29501 --config_file ./accelerate_configs/accelerate_config_vla_1_node.yaml \
    train_vla.py \
    --vlm_name_or_path "/root/hf_models/Qwen/Qwen2.5-VL-3B-Instruct" \
    --FAST_tokenizer_path "/root/hf_models/physical-intelligence/fast" \
    --per_device_train_batch_size 8 \
    --seed 42 \
    --epochs 48 \
    --save_ckpt_interval 16 \
    --save_step_interval 3000 \
    --lr 5e-5 \
    --tensorboard_log_dir "/workspace/zhaochen/reasoningVLA-ruc/outputs/ckpts/train_logs/reasoning_vla-3b-20251219_qwen-real" \
    --output_ckpt_dir "/workspace/zhaochen/reasoningVLA-ruc/outputs/ckpts/reasoning_vla-3b-20251219_qwen-real" \
    --tune_visual \
    --tune_llm \
    --tune_projector \
    --tune_dit \
    --dataset_path_list "/workspace/dataset/ruc_xarm_tabletop/real_world_tabletop_tasks_1013" \
    --mask_prompt_loss \
    --add_discrete_action_tokens \
    --window_size 1 \
    --action_horizon 32 \
    --max_pad_state_and_action_length 64 \
    --desc_type "bpe" \
    --backbone_type "qwen2.5vl"
