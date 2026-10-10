import pickle
import os
import imageio
import pandas as pd
import numpy as np
import json
import pyarrow as pa
import io
import time
import random
import math
import os
import matplotlib.pyplot as plt

from datasets import Dataset, Features, Value, Sequence
from datasets import Image as datasets_Image
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

print("os.cpu_count():", os.cpu_count())

def obtain_stats(data, data_type):
    result = {'count': [len(data)], 'max': [], 'mean': [], 'min': [], 'std': []}

    if data_type == "frames":
        stacked_data = np.stack(data, axis=0)/255 # e.g., (69, 720, 1280, 3)
        calcualte_axis = (0, 1, 2)
    elif data_type in ["state", "action"]:
        stacked_data = np.stack(data, axis=0) # e.g., (69, 8)
        calcualte_axis = 0
    elif data_type == "others":
        stacked_data = np.stack(data, axis=0) # e.g., (69,)
        stacked_data = stacked_data[:, np.newaxis] # e.g., (69, 1)
        calcualte_axis = 0
    else:
        raise ValueError(f"data_type unsupported: {data_type}")

    start_time = time.time()
    # calculate statistics
    max_values = np.max(stacked_data, axis=calcualte_axis).tolist()
    mean_values = np.mean(stacked_data, axis=calcualte_axis).tolist()
    min_values = np.min(stacked_data, axis=calcualte_axis).tolist()
    std_values = np.std(stacked_data, axis=calcualte_axis).tolist()
    # print(time.time() - start_time)

    result = {'count': [len(data)]}
    if data_type == "frames":
        result['max'] = [[[value]] for value in max_values]
        result['mean'] = [[[value]] for value in mean_values]
        result['min'] = [[[value]] for value in min_values]
        result['std'] = [[[value]] for value in std_values]
    elif data_type in ["state", "action", "others"]:
        result['max'] = max_values
        result['mean'] = mean_values
        result['min'] = min_values
        result['std'] = std_values

    # print(result)
    return result

def save_jsonl(file_path, data):
    with open(file_path, 'w') as f:
        for item in data:
            json_line = json.dumps(item)
            f.write(json_line + '\n')

def find_pickle_files(directory):
    pickle_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pickle'):
                pickle_files.append(os.path.join(root, file))
    return pickle_files

def load_image(image_bytes):
    '''
    PIL image bytes -> image (numpy array)
    '''
    img = Image.open(io.BytesIO(image_bytes))
    # print(img)
    arr = np.array(img)

    arr = arr[..., ::-1] # NOTE: [IMPORTANT] convert BGR to RGB
    return arr

if __name__ == "__main__":
    '''
    lerobot dataset (the video version)
    changes compared to the image version:
        1. there are not camera keys (or data) in the parquent
        2. revise info.json file, `total_videos`, `video_path`, and `video_info` in `features`.cameras
    '''
    random.seed(42)
    pickle_files = find_pickle_files("pickle_files/example1")
    random.shuffle(pickle_files)
    print(pickle_files)

    # Each pickle file contains the following keys:
    # dict_keys(['pose', 'joint', 'standard_joint', 'image_wrist', 'image_front', 'image_left', 'natural_language'])
    dataset_dir = "lerobot_dataset_files/example-tabletop-tasks-video-version"

    os.makedirs(os.path.join(dataset_dir, "data", "chunk-000"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "meta"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "videos", "chunk-000"), exist_ok=True)

    print("num of trajectories:", len(pickle_files))
    fps = 5

    tasks = []
    for pickle_file in tqdm(pickle_files):
        with open(pickle_file, 'rb') as file:
            print(pickle_file)
            trajectory = pickle.load(file)
            for frame in trajectory:
                tasks.append(frame["natural_language"])

    tasks = list(dict.fromkeys(tasks)) # deduplicate with order
    task2id = {task: index for index, task in enumerate(tasks)}
    
    print(task2id)
    # save tasks.jsonl
    save_jsonl(
        os.path.join(dataset_dir, "meta", "tasks.jsonl"),
        [{"task_index": task_index, "task": task} for task, task_index in task2id.items()]
    )

    EMBODIMENT_ID = 0

    state_dim = None
    action_dim = None

    episodes = []
    episodes_stats = []
    total_frames = 0
    global_index = 0

    episode_index = 0
    num_idle_frames = 0

    for pickle_file in tqdm(pickle_files, desc="processing pickle files..."):
        start_time = time.time()
        print(pickle_file)
        # each pickle is a complete trajectory (episode)
        with open(pickle_file, 'rb') as file:
            trajectory = pickle.load(file)

        # camera images
        image_wrist_list = []
        image_front_list = []
        image_left_list = []

        state_list = []
        action_list = []
        episode_index_list = []
        valid_frame_index_list = []
        timestamp_list = []
        index_list = []
        task_index_list = []
        tasks_desc_list = []
        embodiment_id_list = []

        valid_frame_index = 0
        for original_frame_index, frame in enumerate(trajectory):
            if original_frame_index >= len(trajectory) - 1: # we do not use the last frame in the trajectory because it does not have labels
                continue

            if frame["pose"] == trajectory[original_frame_index+1]["pose"]:
                print("idle frame")
                num_idle_frames += 1
                continue
            
            image_wrist_list.append(load_image(frame["image_wrist"]))
            image_front_list.append(load_image(frame["image_front"]))
            image_left_list.append(load_image(frame["image_left"]))

            # state is the end-effectors poses of the current frame
            state_list.append(frame["pose"])
            state_dim = len(state_list[-1])

            # action is the end-effectors poses of the next frame
            action_list.append(trajectory[original_frame_index+1]["pose"])
            action_dim = len(action_list[-1])

            episode_index_list.append(episode_index)
            valid_frame_index_list.append(valid_frame_index)
            timestamp_list.append(valid_frame_index/fps)
            index_list.append(global_index)
            global_index += 1

            task_index_list.append(task2id[frame["natural_language"]])
            tasks_desc_list.append(frame["natural_language"])
            
            embodiment_id_list.append(EMBODIMENT_ID)

            total_frames += 1
            valid_frame_index += 1

        if len(valid_frame_index_list) == 0:
            next_done_list = []
            continue
        else:
            next_done_list = [False] * (len(valid_frame_index_list) - 1) + [True]
        
        episodes.append(
            {
                "episode_index": episode_index,
                "tasks": list(dict.fromkeys(tasks_desc_list)),
                "length": len(valid_frame_index_list)
            }
        )

        expected_len = len(valid_frame_index_list)
        try:
            assert all(len(lst) == expected_len for lst in \
                       [state_list, action_list, episode_index_list, valid_frame_index_list, timestamp_list, next_done_list, index_list, task_index_list])
        except AssertionError:
            print("AssertionError! Length of each list:")
            print("len(state_list):", len(state_list))
            print("len(action_list):", len(action_list))
            print("len(episode_index_list):", len(episode_index_list))
            print("len(valid_frame_index_list):", len(valid_frame_index_list))
            print("len(timestamp_list):", len(timestamp_list))
            print("len(next_done_list):", len(next_done_list))
            print("len(index_list):", len(index_list))
            print("len(task_index_list):", len(task_index_list))
            raise   # Continue to throw exceptions and terminate the program
        
        # calculate statistics of state and action
        episodes_stats.append(
            {
                "episode_index": episode_index,
                "stats": {
                    "observation.state": obtain_stats(state_list, "state"),
                    "action": obtain_stats(action_list, "action"),
                }
            }
        )

        data = {
            "observation.state": state_list,
            "action": action_list,
            "episode_index": episode_index_list,
            "frame_index": valid_frame_index_list,
            "timestamp": timestamp_list,
            "next.done": next_done_list,
            "index": index_list,
            "task_index": task_index_list,
            "embodiment_id": embodiment_id_list
        }

        features = Features({
            "observation.state": Sequence(
                feature=Value(dtype="float64"),
                length=-1  # -1 represents variable length
            ),
            "action": Sequence(
                feature=Value(dtype="float64"),
                length=-1
            ),
            "episode_index": Value(dtype="int32"),
            "frame_index": Value(dtype="int32"),
            "timestamp": Value(dtype="float64"),
            "next.done": Value(dtype="bool"),
            "index": Value(dtype="int32"),
            "task_index": Value(dtype="int32"),
            "embodiment_id": Value(dtype="int32")
        })

        df = pd.DataFrame(data)
        dataset = Dataset.from_pandas(df, features=features)
        # print(df)
        
        # save DataFrame as Parquet
        parquet_file_name = f"episode_{episode_index:06d}.parquet"
        dataset.to_parquet(
            os.path.join(dataset_dir, "data", "chunk-000", parquet_file_name),
            # engine='pyarrow', compression='snappy'
        )

        # save image as mp4 video
        video_file_name = f"episode_{episode_index:06d}.mp4"
        frames_and_cameras = [
            [image_wrist_list, "observation.images.cam_wrist"],
            [image_front_list, "observation.images.cam_front"],
            [image_left_list, "observation.images.cam_left"]
        ]
        for frames, cemera in frames_and_cameras:
            os.makedirs(os.path.join(dataset_dir, "videos", "chunk-000", cemera), exist_ok = True)
            video_path = os.path.join(dataset_dir, "videos", "chunk-000", cemera, video_file_name)
            imageio.mimsave(str(video_path), np.stack(frames), fps=fps)
        
        print(f"time: {time.time() - start_time}")
        episode_index += 1

    print("num_idle_frames:", num_idle_frames)
    save_jsonl(os.path.join(dataset_dir, "meta", "episodes_stats.jsonl"), episodes_stats)
    save_jsonl(os.path.join(dataset_dir, "meta", "episodes.jsonl"), episodes)

    video_info = {
        "video.fps": float(fps),
        "video.codec": "libx264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "has_audio": False
    }

    info = {
        "codebase_version": "v2.1",
        "robot_type": "xarm",
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(task2id),
        "total_videos": len(episodes) * 3, # if we have 3 cemeras
        "total_chunks": 1,
        "chunks_size": 1000000,
        "fps": 5,
        "splits": {
            "train": f"0:{len(episodes)}"
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.cam_wrist": {'dtype': 'video', 'shape': image_wrist_list[0].shape, 'names': ['height', 'width', 'channel'], "video_info": video_info},
            "observation.images.cam_front": {'dtype': 'video', 'shape': image_front_list[0].shape, 'names': ['height', 'width', 'channel'], "video_info": video_info},
            "observation.images.cam_left": {'dtype': 'video', 'shape': image_left_list[0].shape, 'names': ['height', 'width', 'channel'], "video_info": video_info},
            "observation.state": {
                "dtype": "float32",
                "shape": [
                    state_dim
                ],
                "names": {
                    "motors": [
                        f"motor_{index}" for index in range(state_dim)
                    ]
                }
            },
            "action": {
                "dtype": "float32",
                "shape": [
                    action_dim
                ],
                "names": {
                    "motors": [
                        f"motor_{index}" for index in range(action_dim)
                    ]
                }
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [
                    1
                ],
                "names": None
            },
            "frame_index": {
                "dtype": "int64",
                "shape": [
                    1
                ],
                "names": None
            },
            "timestamp": {
                "dtype": "float32",
                "shape": [
                    1
                ],
                "names": None
            },
            "next.done": {
                "dtype": "bool",
                "shape": [
                    1
                ],
                "names": None
            },
            "index": {
                "dtype": "int64",
                "shape": [
                    1
                ],
                "names": None
            },
            "task_index": {
                "dtype": "int64",
                "shape": [
                    1
                ],
                "names": None
            },
            "embodiment_id": {
                "dtype": "int64",
                "shape": [
                    1
                ],
                "names": None
            }
        }
    }

    with open(os.path.join(dataset_dir, "meta", "info.json"), "w") as f:
        f.write(json.dumps(info, indent = 4, ensure_ascii = False))