# constants.py
from utils.normalization import xarm_tabletop_post_action_denorm, libero_post_action_denorm, standard_post_action_denorm

ENV_TYPE2FEATURE = {
    "libero": {
        # "dataset_dir": "/workspace/dataset/sim_benchmarks/libero_v21",
        "dataset_dir": "/workspace/zhaochen/lerobot_dataset/libero_backup",
        "embodiment_id": 45,
        "action_dim" : 7,
        "action_denorm_func": libero_post_action_denorm
    },
    "real_world_xarm_tabletop": {
        "dataset_dir": "/pretrain_data/real_tabletop_tasks_long_cot_lerobot-20250729",
        "embodiment_id": 0,
        "action_dim" : 10,
        "action_denorm_func": xarm_tabletop_post_action_denorm,
    },
    "robotwin2.0-aloha-agilex-2x50tasks": {
        "dataset_dir": "/workspace/dataset/sim_benchmarks/robotwin2.0-aloha-agilex_clean_2x50tasks",
        "embodiment_id": 46,
        "action_dim" : 14,
        "action_denorm_func": standard_post_action_denorm
    },
    "split_1117": {
        "dataset_dir": "/workspace/zhaochen/.lerobot_cache/lerobot/split_dataset_1117_cot",
        "embodiment_id": 46,
        "action_dim" : 14,
        "action_denorm_func": standard_post_action_denorm
    },
    "split_1114": {
        "dataset_dir": "/workspace/zhaochen/.lerobot_cache/lerobot/split_dataset_1114",
        "embodiment_id": 46,
        "action_dim" : 14,
        "action_denorm_func": standard_post_action_denorm
    },
    "split_1124": {
        "dataset_dir": "/workspace/zhaochen/.lerobot_cache/lerobot/split_dataset_1124",
        "embodiment_id": 46,
        "action_dim" : 14,
        "action_denorm_func": standard_post_action_denorm
    },
    "split_1201": {
        "dataset_dir": "/workspace/zhaochen/.lerobot_cache/lerobot/split_dataset_1201",
        "embodiment_id": 46,
        "action_dim" : 14,
        "action_denorm_func": standard_post_action_denorm
    },
    "split_1208": {
        "dataset_dir": "/workspace/zhaochen/.lerobot_cache/lerobot/split_dataset_1208",
        "embodiment_id": 46,
        "action_dim" : 14,
        "action_denorm_func": standard_post_action_denorm
    },
    "split_1209": {
        "dataset_dir": "/workspace/zhaochen/.lerobot_cache/lerobot/split_dataset_1209",
        "embodiment_id": 46,
        "action_dim" : 14,
        "action_denorm_func": standard_post_action_denorm
    },
    "split_1217": {
        "dataset_dir": "/workspace/zhaochen/.lerobot_cache/lerobot/split_dataset_1217",
        "embodiment_id": 46,
        "action_dim" : 14,
        "action_denorm_func": standard_post_action_denorm
    },
    "split_1219": {
        "dataset_dir": "/workspace/zhaochen/.lerobot_cache/lerobot/split_dataset_1219",
        "embodiment_id": 46,
        "action_dim" : 14,
        "action_denorm_func": standard_post_action_denorm
    },
    "split_1223": {
        "dataset_dir": "/workspace/zhaochen/.lerobot_cache/lerobot/split_dataset_1223",
        "embodiment_id": 46,
        "action_dim" : 14,
        "action_denorm_func": standard_post_action_denorm
    },
    "split_1226": {
        "dataset_dir": "/workspace/zhaochen/.lerobot_cache/lerobot/split_dataset_1226",
        "embodiment_id": 46,
        "action_dim" : 14,
        "action_denorm_func": standard_post_action_denorm
    },
}
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
QUAD_START_TOKEN = '<quad>'
QUAD_END_TOKEN = '</quad>'
REF_START_TOKEN = '<ref>'
REF_END_TOKEN = '</ref>'
BOX_START_TOKEN = '<box>'
BOX_END_TOKEN = '</box>'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
CLIP_STD = (0.2686295, 0.2613025, 0.2757711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)