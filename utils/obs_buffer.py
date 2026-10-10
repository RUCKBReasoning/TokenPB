import os
import time
import torch
from collections import deque
from torchvision.transforms import ToPILImage

class Observation:
    def __init__(self, multi_view_images: dict):
        self.multi_view_images = multi_view_images

class ObservationBuffer:
    def __init__(self, max_recent_observations: int, remain_ep_start: bool):
        self.max_recent_observations = max_recent_observations
        self.remain_ep_start = remain_ep_start

        self.ep_start_observation = None
        self.recent_observations = deque(maxlen = self.max_recent_observations)

        if self.remain_ep_start:
            self.inference_required_ob_num = self.max_recent_observations + 1
        else:
            self.inference_required_ob_num = self.max_recent_observations 

    def add_observation(self, is_ep_start: bool, multi_view_images: dict):
        ob = Observation(multi_view_images = multi_view_images)
        if is_ep_start and self.remain_ep_start:
            self.ep_start_observation = ob
        else:
            self.recent_observations.append(ob)
    
    def reset(self):
        self.ep_start_observation = None
        self.recent_observations = deque(maxlen = self.max_recent_observations)

    def aggregate_observations(self, observations: list[Observation], camera_keys: dict):
        agg_observations = dict()
        for camera_key in camera_keys:
            agg_observations[camera_key] = []

        for ob in observations:
            for camera_key in camera_keys:
                agg_observations[camera_key].append(ob.multi_view_images[camera_key])
        
        for key in agg_observations.keys():
            agg_observations[key] = torch.stack(agg_observations[key], dim=0)
            # print(f"{key}.shape=", agg_observations[key].shape)
        # print("-"*30)

        return agg_observations

    def get_inference_time_observations(self, camera_keys: dict, visualize: bool = False):
        inference_time_observations = []
        
        if self.remain_ep_start:
            inference_time_observations.append(self.ep_start_observation)
        
        for recent_ob in list(self.recent_observations):
            inference_time_observations.append(recent_ob)

        if len(inference_time_observations) < self.inference_required_ob_num:
            # left pad with the earliest observations
            pad_observations = [inference_time_observations[0]] * (self.inference_required_ob_num - len(inference_time_observations))
            inference_time_observations = pad_observations + inference_time_observations

        agg_observations = self.aggregate_observations(inference_time_observations, camera_keys)

        if visualize:
            timestamp = str(int(time.time()))            
            for camera_key in camera_keys:
                os.makedirs(os.path.join('temp', timestamp, camera_key), exist_ok=True)
                for img_idx, torch_image in enumerate(agg_observations[camera_key]):
                    to_pil = ToPILImage()
                    pil_image = to_pil(torch_image)
                    pil_image.save(os.path.join("temp", timestamp, camera_key, f"{img_idx}.jpg"))

        return agg_observations