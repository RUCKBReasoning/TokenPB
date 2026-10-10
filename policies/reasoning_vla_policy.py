import torch
import numpy as np

from typing import Dict
from torchvision.transforms import ToTensor
from transformers import Qwen2_5_VLProcessor, AutoProcessor
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from model import ReasoningVLAModel
from utils.normalization import rotation_norm
from utils.load_training_dataset import prepare_qwen2_5_vl_inputs_cpu, prepare_internvl3_5_inputs_cpu, prepare_action_expert_inputs_cpu, custom_collate_fn
from utils.obs_buffer import ObservationBuffer
from utils.constants import ENV_TYPE2FEATURE
from policies.base_policy import BasePolicy

class ReasoningVLAPolicy(BasePolicy):
    def __init__(
        self,
        env_type,
        ckpt_dir,
        window_size,
        remain_ep_start,
        num_denoised_steps=5,
        max_pad_state_and_action_length=64,
        backbone = "qwen2.5vl",
        device="cuda:0"
    ):
        super().__init__()
        # env params
        self.env_type = env_type
        self.ckpt_dir = ckpt_dir
        self.window_size = window_size
        self.remain_ep_start = remain_ep_start
        self.num_denoised_steps = num_denoised_steps
        self.max_pad_state_and_action_length = max_pad_state_and_action_length
        self.device = device
        self.backbone = backbone
        self.fast_tokenizer = AutoProcessor.from_pretrained("/workspace/zhaochen/reasoningVLA-ruc/outputs/fast", trust_remote_code=True)

        

        # load relevant configurations from the constant pool
        env_cfg = ENV_TYPE2FEATURE[env_type]
        self.dataset_dir = env_cfg["dataset_dir"]
        self.embodiment_id = env_cfg["embodiment_id"]
        self.action_dim = env_cfg["action_dim"]
        self.action_denorm_func = env_cfg["action_denorm_func"]

        # load VLM's processor
        if self.backbone == "qwen2.5vl":
            self.processor = Qwen2_5_VLProcessor.from_pretrained(ckpt_dir)
        else:
            self.processor = AutoProcessor.from_pretrained(ckpt_dir)
        # load dataset's metadata
        self.dataset_meta = LeRobotDatasetMetadata(
            repo_id = self.dataset_dir.split("/")[-1],
            root = self.dataset_dir
        )
        # load model
        self.model = ReasoningVLAModel.from_pretrained(ckpt_dir, backbone=self.backbone).to(device).to(torch.bfloat16)
        self.model.eval()
        # initialize buffer
        self.ob_buffer = ObservationBuffer(max_recent_observations=window_size, remain_ep_start=remain_ep_start)
        self.to_tensor = ToTensor()
        self.global_inference_steps = 0
        self.action_batch_list = []

    def infer(self, obs: Dict) -> Dict:
        # obs: {"task": ..., "observation.state": ..., "n_action_steps": ..., "is_ep_start": ..., <cam_keys>: ...}

        task = obs.get('task')
        state = obs.get('observation.state')
        n_action_steps = obs.get('n_action_steps')
        is_ep_start = obs.get('is_ep_start')

        if self.env_type == "real_world_xarm_tabletop":
            if len(state) == 7:  # remove the gripper's state 
                state = state[:-1]
            state = rotation_norm(state) # xyzrpy -> xyzsin(r)cos(r)...

        # convert image from numpy to tensor
        multi_view_images = dict()
        for cam_key in self.dataset_meta.camera_keys:
            multi_view_images[cam_key] = self.to_tensor(np.array(obs.get(cam_key), dtype=np.uint8))

        # manage historical observation buffer
        if is_ep_start:
            self.ob_buffer.reset()
            self.global_inference_steps = 0
        self.ob_buffer.add_observation(is_ep_start, multi_view_images)
        
        # get the latest observations from the buffer
        observations = self.ob_buffer.get_inference_time_observations(self.dataset_meta.camera_keys, visualize = False)
        
        print("task:", task)
        data_sample = {
            "task": task, "embodiment_id": self.embodiment_id, "observation.state": state
        }
        for key, value in observations.items():
            data_sample[key] = value
        
        action_inputs = prepare_action_expert_inputs_cpu(
            data_sample, self.dataset_meta.stats, self.max_pad_state_and_action_length, action_horizon=32
        )
        if self.backbone == "qwen2.5vl":
            vl_inputs = prepare_qwen2_5_vl_inputs_cpu(
                data=data_sample,
                stat = self.dataset_meta.stats,
                camera_keys=self.dataset_meta.camera_keys,
                grounding_camera_keys=self.dataset_meta.grounding_camera_keys,
                processor=self.processor,
                process_mode="eval",
                # `add_discrete_action_tokens`, `fast_tokenizer`, and `mask_prompt_loss` are not used during infer
            )
        elif self.backbone=="internvl3.5":
            vl_inputs = prepare_internvl3_5_inputs_cpu(
                data=data_sample,
                stat = self.dataset_meta.stats,
                camera_keys=self.dataset_meta.camera_keys,
                grounding_camera_keys=self.dataset_meta.grounding_camera_keys,
                processor=self.processor,
                process_mode="eval",
                # `add_discrete_action_tokens`, `fast_tokenizer`, and `mask_prompt_loss` are not used during infer
            )
        else:
            raise ValueError(f"could not find backbone key {self.backbone}.")
        
        # prepare batch input
        vla_input = custom_collate_fn([{**action_inputs, **vl_inputs}])
        for key, value in vla_input.items():
            vla_input[key] = value.to(self.device)

        # model inference
        with torch.no_grad():
            vla_outputs = self.model.get_action(vla_input, self.num_denoised_steps, action_dim=self.action_dim, action_batch_list=self.action_batch_list, action_tokenizer = self.fast_tokenizer)
        '''
        1. bs=1
        2. only retain the next `n_action_steps` steps in the chunk
        3. slice out the first few `action_dim` (the remaining dim are padded with 0)
        '''
        self.action_batch_list = vla_outputs['candicate_list']
        action_chunk = vla_outputs["action_pred"][0][:n_action_steps, :self.action_dim]
        # action_chunk = vla_outputs["action_pred"]
        action_chunk = self.action_denorm_func(action_chunk, self.dataset_meta)
        if vla_outputs["modify_action"] is not None:
            modify_action = self.action_denorm_func(vla_outputs["modify_action"][:n_action_steps, :self.action_dim], self.dataset_meta)
            # action = np.array(action_chunk)
            # modify = np.array(modify_action)
            # w1, w2 = 0, 1
            # result_array = np.concatenate((action[:8], w1 * action[8:16] + w2 * modify[8:16], w1 * action[16:]))
            # action_chunk = result_array.tolist()
        else:
            modify_action = None
        self.global_inference_steps += 1
        print(action_chunk)
        print(modify_action)

        return {"actions": action_chunk}