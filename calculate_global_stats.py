import pandas as pd
import glob
from tqdm import tqdm
import numpy as np
import json

'''
This script aims to calculate the global statistics of actions and states
'''

def get_stats(arr):
    # arr shape: (N, dim)
    mean = np.mean(arr, axis=0)
    std  = np.std(arr, axis=0)
    min_ = np.min(arr, axis=0)
    max_ = np.max(arr, axis=0)
    q01  = np.percentile(arr, 1, axis=0)
    q99  = np.percentile(arr, 99, axis=0)
    return {
        'mean':   mean.tolist(),
        'std':    std.tolist(),
        'min':    min_.tolist(),
        'max':    max_.tolist(),
        'q01':    q01.tolist(),
        'q99':    q99.tolist()
    }

# dataset_dirs = [
#     # "/pretrain_data/AgiBotAlpha_covt2lerobot_0818",
#     # "/pretrain_data/AgiBotBeta_covt2lerobot_0818", # has errors (action dim=6)
#     "/pretrain_data/rh20t/RH20T_cfg1_lerobot",
#     "/pretrain_data/rh20t/RH20T_cfg2_lerobot",
#     "/pretrain_data/rh20t/RH20T_cfg3_lerobot",
#     "/pretrain_data/rh20t/RH20T_cfg4_lerobot",
#     "/pretrain_data/rh20t/RH20T_cfg6_lerobot",
#     "/pretrain_data/rh20t/RH20T_cfg7_lerobot",
#     "/pretrain_data/oxe_lerobot/bridge_data_v2_new",
#     "/pretrain_data/oxe_lerobot/io_ai_tech",
#     "/pretrain_data/oxe_lerobot/jaco_play",
#     "/pretrain_data/oxe_lerobot/austin_buds_dataset_converted_externally_to_rlds",
#     "/pretrain_data/oxe_lerobot/robo_set_new",
#     "/pretrain_data/oxe_lerobot/cmu_play_fusion",
#     "/pretrain_data/oxe_lerobot/fractal20220817_data",
#     "/pretrain_data/oxe_lerobot/iamlab_cmu_pickup_insert_converted_externally_to_rlds",
#     "/pretrain_data/oxe_lerobot/plex_robosuite",
#     "/pretrain_data/oxe_lerobot/dobbe",
#     "/pretrain_data/oxe_lerobot/ucsd_kitchen_dataset_converted_externally_to_rlds",
#     "/pretrain_data/oxe_lerobot/nyu_door_opening_surprising_effectiveness",
#     "/pretrain_data/oxe_lerobot/toto",
#     "/pretrain_data/oxe_lerobot/stanford_hydra_dataset_converted_externally_to_rlds",
#     "/pretrain_data/oxe_lerobot/utaustin_mutex",
#     "/pretrain_data/oxe_lerobot/ucsd_pick_and_place_dataset_converted_externally_to_rlds",
#     "/pretrain_data/oxe_lerobot/columbia_cairlab_pusht_real",
#     "/pretrain_data/oxe_lerobot/cmu_stretch",
#     "/pretrain_data/oxe_lerobot/berkeley_fanuc_manipulation",
#     "/pretrain_data/oxe_lerobot/dlr_edan_shared_control_converted_externally_to_rlds",
#     "/pretrain_data/oxe_lerobot/dlr_sara_pour_converted_externally_to_rlds",
#     "/pretrain_data/oxe_lerobot/berkeley_autolab_ur5",
#     "/pretrain_data/oxe_lerobot/bridge",
#     "/pretrain_data/oxe_lerobot/qut_dexterous_manpulation",
#     "/pretrain_data/oxe_lerobot/austin_sailor_dataset_converted_externally_to_rlds",
#     "/pretrain_data/oxe_lerobot/austin_sirius_dataset_converted_externally_to_rlds",
#     "/pretrain_data/oxe_lerobot/nyu_franka_play_dataset_converted_externally_to_rlds",
#     "/pretrain_data/oxe_lerobot/roboturk",
#     "/pretrain_data/oxe_lerobot/taco_play",
#     "/pretrain_data/oxe_lerobot/bc_z_new1",
#     "/pretrain_data/oxe_lerobot/robo_net",
#     "/pretrain_data/oxe_lerobot/uiuc_d3field",
#     "/pretrain_data/oxe_lerobot/dlr_sara_grid_clamp_converted_externally_to_rlds",
#     "/pretrain_data/oxe_lerobot/language_table",
#     "/pretrain_data/oxe_lerobot/fmb"
# ]


dataset_dirs = [
    "/workspace/zhaochen/.lerobot_cache/lerobot/split_dataset_1223"
]

if __name__ == "__main__":
    for idx, dataset_dir in enumerate(dataset_dirs):
        print(idx, ":", dataset_dir)
        state_list = []
        action_list = []
        
        parquet_files = glob.glob(f"{dataset_dir}/data/**/*.parquet", recursive=True)
        # print(parquet_files)

        for p_file in tqdm(parquet_files):
            df = pd.read_parquet(p_file)

            state = np.stack(df['observation.state'].values)
            action = np.stack(df['action'].values)

            state_list.append(state)
            action_list.append(action)

        all_states = np.concatenate(state_list, axis=0)   # shape (N, state_dim)
        all_actions = np.concatenate(action_list, axis=0) # shape (N, action_dim)

        stats = {
            "observation.state": get_stats(all_states),
            "action": get_stats(all_actions)
        }

        with open(f"{dataset_dir}/meta/stats.json", "w") as f:
            json.dump(stats, f, indent=4)