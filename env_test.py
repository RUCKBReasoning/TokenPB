from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import pickle
import os

lerobot_data_path = "example-tabletop-tasks-video-version"
lerobot_dataset = LeRobotDataset(
    repo_id = lerobot_data_path.split("/")[-1],
    root = lerobot_data_path
)
print(lerobot_dataset[0])

file_path = '/workspace/pretrain_data/lihaoyang/reasoningVLA/pickle_files/data1747638980.1726224.data.pickle'
# 以二进制模式打开文件并加载数据
with open(file_path, 'rb') as file:
    data = pickle.load(file)
    print(data)
