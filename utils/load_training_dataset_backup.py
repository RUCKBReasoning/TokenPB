import torch
import hashlib
import time
import torch.distributed as dist
import random

import numpy as np
from torchvision.transforms import ToPILImage
from transformers import Qwen2_5_VLProcessor, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils.normalization import min_max_norm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from torch.utils.data import ConcatDataset, DataLoader

def pad_2d_to_max_length(tensor_2d, max_pad_length=64, pad_value=0.0):
    """
    Pads each row of a 2D tensor to max_length with pad_value and returns the corresponding mask.
    
    Args:
        tensor_2d (torch.Tensor): shape (N, L), L <= max_length
        max_length (int): pad length
        pad_value (int or float): value for padding
    
    Returns:
        padded_tensor (torch.Tensor): shape (N, max_length)
        mask (torch.Tensor): shape (N, max_length), 1=real, 0=pad
    """
    N, L = tensor_2d.shape
    if L >= max_pad_length:
        return tensor_2d[:, :max_pad_length], torch.ones((N, max_pad_length), dtype=torch.long, device=tensor_2d.device)
    pad_len = max_pad_length - L
    pad_tensor = torch.full((N, pad_len), pad_value, dtype=tensor_2d.dtype, device=tensor_2d.device)
    padded_tensor = torch.cat([tensor_2d, pad_tensor], dim=1)
    mask = torch.cat([
        torch.ones((N, L), dtype=torch.long, device=tensor_2d.device),
        torch.zeros((N, pad_len), dtype=torch.long, device=tensor_2d.device)
    ], dim=1)
    return padded_tensor, mask

def prepare_action_expert_inputs_cpu(data, stats, max_pad_length, action_horizon):
    action_expert_inputs = {}
    # a scalar (shape is [])
    if not isinstance(data["embodiment_id"], torch.Tensor):
        data["embodiment_id"] = torch.tensor(data["embodiment_id"])
    action_expert_inputs["embodiment_id"] = data["embodiment_id"].to(torch.long)

    # 1. per-dim min-max norm; 2. dim-size padding + return mask
    padded_states, state_masks = pad_2d_to_max_length(
        min_max_norm(data["observation.state"], stats["observation.state"]), max_pad_length
    )
    # NOTE: state contains only 1 token
    action_expert_inputs["observation.state"] = padded_states  # (1, max_length)
    action_expert_inputs["state_mask"] = state_masks           # (1, max_length)

    if "action" in data:
        padded_actions, action_masks = pad_2d_to_max_length(
            min_max_norm(data["action"], stats["action"]), max_pad_length
        )
        # NOTE: action contains `action_horizon` tokens
        action_expert_inputs["action"] = padded_actions    # (action_horizon, max_length)
        action_expert_inputs["action_mask"] = action_masks # (action_horizon, max_length)
    else:
        real_action_dim = len(stats["action"]["q01"])
        mask = torch.zeros((action_horizon, max_pad_length), dtype=torch.int32)
        mask[:, :real_action_dim] = 1
        action_expert_inputs["infer_action_mask"] = mask # (action_horizon, max_length)

    return action_expert_inputs

def find_subtensor_index(tensor, sub_tensor):
    """
    Find the first occurrence of the 1D sub-string sub_tensor in the 1D tensor tensor, 
        and return -1 if it is not found
    """
    n = sub_tensor.numel()
    if n == 0 or n > tensor.numel():
        return -1
    # create sliding windows of length n on the original tensor
    windows = tensor.unfold(0, n, 1)          # [L-n+1, n]
    # compare each window with sub_tensor
    matches = (windows == sub_tensor).all(dim=1)
    # find indices where a full match occurs
    idxs = torch.where(matches)[0]
    return idxs[0].item() if idxs.numel() > 0 else -1

def mask_prompt_loss_1d(labels_1d: torch.Tensor, assistant_start_ids, pad_token_id):
    """
    Mask a single sample (1D) labels:
    - Set all pad_token_ids to -100.
    - Set the content before the assistant starting tokens (i.e., the generation prompt) to -100. (The starting tokens themselves will not be masked to -100.)
    """
    # pad -> -100
    if pad_token_id is not None:
        labels_1d[labels_1d == pad_token_id] = -100
    # mask prompt（assistant starting token sequence: [151644, 77091, 198]）
    ast = torch.tensor(list(assistant_start_ids), dtype=labels_1d.dtype, device=labels_1d.device)
    start_idx = find_subtensor_index(labels_1d, ast)
    if start_idx != -1:
        labels_1d[:start_idx] = -100
    return labels_1d

def _trans_action_to_token(action: torch.Tensor, n_bins: int = 256):
    bins = np.linspace(-1, 1, n_bins)
    action = np.clip(action, a_min=-1., a_max=1.)
    discretized_action = np.digitize(action, bins)
    return discretized_action

def prepare_qwen2_5_vl_inputs_cpu(
    data: dict,
    stat,
    camera_keys: list[str],
    grounding_camera_keys: list[str],
    processor: Qwen2_5_VLProcessor,
    process_mode: str,
    add_discrete_action_tokens: bool = False,
    fast_tokenizer: AutoProcessor = None,
    mask_prompt_loss: bool = True,
    desc_type = "bpe"
):
    to_pil = ToPILImage()
    max_length = 2048
    resized_height = 224
    resized_width = 224

    # store contextual images (history + current)
    context_images = []
    if len(data[camera_keys[0]].shape) == 4:
        # for each camera, the lerobot dataset gives several historical observations and one current observations
        # For example, torch.Size([4, 3, 480, 640]) means that it contains 3 historical obs and 1 current obs
        obs_num = data[camera_keys[0]].shape[0]
        for obs_idx in range(obs_num):
            for camera_key in camera_keys:
                cam_key = f"current.{camera_key}" if obs_idx == obs_num - 1 else f"historical-{obs_idx}.{camera_key}"
                context_images.append([cam_key, to_pil(data[camera_key][obs_idx])])
    elif len(data[camera_keys[0]].shape) == 3:
        # for each camera, the lerobot dataset only gives one current observation
        for camera_key in camera_keys:
            context_images.append([f"current.{camera_key}", to_pil(data[camera_key])])
    else:
        raise ValueError(
            f"Expected image shape with 3 or 4 dimensions for data['{camera_keys[0]}'], "
            f"but got shape with {len(data[camera_keys[0]].shape)} dimensions: {data[camera_keys[0]].shape}."
        )

    # grounding camera keys
    if grounding_camera_keys is None:
        grounding_camera_keys = camera_keys[-1]
    grounding_camera_keys_fmt = [f"current.{key}" for key in grounding_camera_keys]
    grounding_camera_keys_str = "[" + ", ".join(grounding_camera_keys_fmt) + "]"

    # prompt is the concat of [str('historical-0.camera1'), img1_0, str('historical-0.camera2'), img2_0, ..., str('historical-1.camera1'), img1_1, ..., str('current.camera1'), img1_n, task_desc)]
    prompt_suffix = f"\nLet's think step by step. Ground task-related objects on the images {grounding_camera_keys_str}."
    prompt = {"role": "user", "content": []}
    for cam_key, img in context_images:
        prompt["content"].append({"type": "text", "text": cam_key})
        prompt["content"].append({"type": "image", "image": img, "resized_height": resized_height, "resized_width": resized_width})
    prompt["content"].append({"type": "text", "text": "Task:\n" + data["task"].strip() + prompt_suffix})
    msg = [prompt]

    if process_mode == "train":
        output_seq = data["reasoning_output"]
        if add_discrete_action_tokens: # add discrete action tokens in VLM's output sequences
            if desc_type == "bins":
                discrete_action_tokens = _trans_action_to_token(data["action"])
                discrete_action_tokens = torch.tensor(discrete_action_tokens).view(-1)
                qwen_discrete_action_tokens = processor.tokenizer.decode(discrete_action_tokens)
                output_seq += "<action>\n" + qwen_discrete_action_tokens + "\n</action>"
            elif desc_type == "qwen":
                discrete_action_tokens = fast_tokenizer(min_max_norm(data["action"], stat["action"]))
                qwen_discrete_action_tokens = processor.tokenizer.decode(discrete_action_tokens[0])
                output_seq += "<action>\n" + qwen_discrete_action_tokens + "\n</action>"
            elif desc_type == "bpe":
                discrete_action_tokens = fast_tokenizer(min_max_norm(data["action"], stat["action"]))
                output_seq += "<action>\n" + str(discrete_action_tokens) + "\n</action>"
            else:
                raise ValueError("Unsupported type: ", desc_type)
            # print("qwen_discrete_action_tokens: ", qwen_discrete_action_tokens)
            # output_seq += "<action>\n" + str(discrete_action_tokens) + "\n</action>"
            if random.random() < 0.001:
                print(data['action'])
                print(data['action'].shape)
                print(output_seq)
            
        msg.append({"role": "assistant", "content": output_seq})

    # for inference, the generation prompt should be added.
    add_generation_prompt = (process_mode != "train")
    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=add_generation_prompt)

    # vision input
    image_inputs, video_inputs = process_vision_info(msg)

    st = time.time()
    if process_mode == "train":
        processor.tokenizer.padding_side = "right"
        vl_inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding='max_length',
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            padding_side="right"
        )
    else:
        # don't perform padding during evaluation
        vl_inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        print(f"qwen2.5 vl's processor spends {time.time()-st}s")
    
    # remove the batch dim of input_ids and attention_mask
    vl_inputs["input_ids"] = vl_inputs["input_ids"][0] # (seq_len,)
    vl_inputs["attention_mask"] = vl_inputs["attention_mask"][0] # (seq_len,)
    # pixel_values and image_grid_thw do not contain the batch dim
    vl_inputs["pixel_values"] = vl_inputs["pixel_values"] # (pixel_values_width, pixel_values_height)
    vl_inputs["image_grid_thw"] = vl_inputs["image_grid_thw"] # (image_grid_thw_width, image_grid_thw_height)

    if process_mode == "train":
        vl_inputs["labels"] = vl_inputs["input_ids"].clone() # (seq_len,)

        if mask_prompt_loss:
            # 151643 is the default pad id of Qwen2.5
            pad_id = getattr(processor.tokenizer, "pad_token_id", 151643)
            masked_labels = mask_prompt_loss_1d(
                vl_inputs["labels"],
                assistant_start_ids=(151644, 77091, 198), # generation prompt: <|im_start|>assistant\n
                pad_token_id=pad_id
            )
            vl_inputs["labels"] = masked_labels

    return vl_inputs

def deterministic_n_action_steps(action_horizon: int, ds_id: int, local_idx: int, epoch: int) -> int:
    # Avoiding the random salt effects of Python's default hash with stable hashing
    key = f"{ds_id}-{local_idx}-{epoch}".encode("utf-8")
    h = hashlib.sha1(key).hexdigest()
    # Take the first 8 bytes as an integer
    val = int(h[:8], 16)
    return 1 + (val % action_horizon)

class StreamingLeRobotSampleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        lerobot_dataset: LeRobotDataset, # a LeRobot Dataset instance
        processor: Qwen2_5_VLProcessor,
        fast_tokenizer: AutoProcessor,
        window_size: int,
        action_horizon: int,
        remain_ep_start_observations: bool,
        process_mode: str = "train",
        add_discrete_action_tokens: bool = False,
        dataset_id: int = 0,
        mask_prompt_loss: int = True,
        max_pad_state_and_action_length: int = 64,
        desc_type = "bpe",
    ):
        self.ds = lerobot_dataset
        self.processor = processor
        self.fast_tokenizer = fast_tokenizer
        self.window_size = max(1, window_size)
        self.action_horizon = action_horizon
        self.remain_ep_start_observations = remain_ep_start_observations
        self.process_mode = process_mode
        self.add_discrete_action_tokens = add_discrete_action_tokens
        self.dataset_id = dataset_id
        self.mask_prompt_loss = mask_prompt_loss
        self.max_pad_state_and_action_length = max_pad_state_and_action_length

        self.fps = self.ds.meta.fps
        self.camera_keys = self.ds.meta.camera_keys
        self.stats = self.ds.meta.stats
        self.grounding_camera_keys = self.ds.meta.grounding_camera_keys
        self.desc_type = desc_type

        # Record the starting global index of the episode each sample belongs to (enabled only if needed)
        if self.remain_ep_start_observations:
            self.global_data_index2ep_start_index = {}
            for global_data_index, episode_index in zip(self.ds.index, self.ds.episode_indices):
                gi = int(global_data_index)
                ep_start_index = self.ds.episode_data_index["from"][episode_index].item()
                self.global_data_index2ep_start_index[gi] = ep_start_index
        else:
            self.global_data_index2ep_start_index = None

        self._epoch = 0  # used for deterministic randomness

    def __len__(self):
        return len(self.ds)

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _obtain_delta_timestamps(self, current_index: int, n_action_steps: int):
        delta_timestamps = dict()
        if self.remain_ep_start_observations:
            ep_start_index = self.global_data_index2ep_start_index[current_index]
            to_ep_start_delta_timestamp = [(ep_start_index - current_index) / self.fps]
        else:
            to_ep_start_delta_timestamp = []

        # During inference, we set `n_action_steps` to an integer between [1, action_horizon]. 
        # Thus, during training, the observation images in the prompt should come from the delta indices: [-(N-1)*n_action_steps, ..., -1*n_action_steps, 0],
        # where N is `window_size` and `n_action_steps` is a random integer between 1 and action_horizon
        slide_window_observation_delta_indices = list(range(self.window_size))[::-1]
        slide_window_observation_delta_indices = [-idx * n_action_steps for idx in slide_window_observation_delta_indices]

        if self.remain_ep_start_observations or self.window_size > 1:
            # set camera's delta timestamps
            for camera_key in self.camera_keys:
                delta_timestamps[camera_key] = to_ep_start_delta_timestamp + \
                    [index / self.fps for index in slide_window_observation_delta_indices]

        # set action's delta timestamps
        action_delta_indices = list(range(self.action_horizon))
        delta_timestamps["action"] = [idx / self.fps for idx in action_delta_indices]
        return delta_timestamps

    def __getitem__(self, idx: int):
        # sample a `n_action_steps` value for current data (epoch stable)
        n_action_steps = deterministic_n_action_steps(
            self.action_horizon, ds_id=self.dataset_id, local_idx=idx, epoch=self._epoch
        )
        delta_timestamps = self._obtain_delta_timestamps(idx, n_action_steps)

        # fetch raw data from Lerobot Dataset
        data_sample = self.ds.getitem_with_delta_timestamps(idx, delta_timestamps)

        # online pre-processing
        action_inputs = prepare_action_expert_inputs_cpu(data_sample, self.stats, self.max_pad_state_and_action_length, self.action_horizon)
        vl_inputs = prepare_qwen2_5_vl_inputs_cpu(
            data=data_sample,
            stat = self.stats,
            camera_keys=self.camera_keys,
            grounding_camera_keys=self.grounding_camera_keys,
            processor=self.processor,
            process_mode=self.process_mode,
            add_discrete_action_tokens=self.add_discrete_action_tokens,
            fast_tokenizer=self.fast_tokenizer,
            mask_prompt_loss=self.mask_prompt_loss,
            desc_type=self.desc_type
        )

        preprocessed_data_sample = {**action_inputs, **vl_inputs}
        
        return preprocessed_data_sample

def build_concat_streaming_dataset(
    lerobot_data_paths: list[str],
    model_name_or_path: str,
    fast_tokenizer_path: str,
    window_size: int,
    action_horizon: int,
    remain_ep_start_observations: bool,
    process_mode: str = "train",
    add_discrete_action_tokens: bool = True,
    mask_prompt_loss: bool = True,
    max_pad_state_and_action_length: int = 64,
    desc_type = "bpe"
):
    lerobot_data_paths = sorted(lerobot_data_paths)
    processor = Qwen2_5_VLProcessor.from_pretrained(model_name_or_path)
    fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)

    datasets = []
    for dataset_id, p in enumerate(lerobot_data_paths):
        base = LeRobotDataset(repo_id=p.split("/")[-1], root=p)
        ds = StreamingLeRobotSampleDataset(
            lerobot_dataset=base,
            processor=processor,
            fast_tokenizer=fast_tokenizer,
            window_size=window_size,
            action_horizon=action_horizon,
            remain_ep_start_observations=remain_ep_start_observations,
            process_mode=process_mode,
            add_discrete_action_tokens=add_discrete_action_tokens,
            dataset_id=dataset_id,
            mask_prompt_loss=mask_prompt_loss,
            max_pad_state_and_action_length=max_pad_state_and_action_length,
            desc_type=desc_type
        )
        datasets.append(ds)

    concat = ConcatDataset(datasets)
    return concat

def custom_collate_fn(batch):
    """
    Generic collate_fn that stacks all keys except for certain keys,
    for which it uses cat along the first dimension.
    """
    cat_keys = ['pixel_values', 'image_grid_thw']  # keys required to perform concat
    keys = batch[0].keys()
    result = {}
    for key in keys:
        items = [item[key] for item in batch]
        if key in cat_keys:
            # Concatenate along the first dimension (does not add a new dimension)
            result[key] = torch.cat(items, dim=0)
        else:
            # Stack items along a new batch dimension (adds a new dimension at dim=0)
            result[key] = torch.stack(items, dim=0)
    
    return result

def create_dataloader_for_concat(
    concat_dataset,
    batch_size_per_device: int,
    num_workers: int = 16,
    prefetch_factor: int = 4
):
    # world_size = dist.get_world_size() if dist.is_initialized() else 1
    # rank = dist.get_rank() if dist.is_initialized() else 0
    # sampler = DistributedSampler(
    #     concat_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
    # )
    dataloader = DataLoader(
        concat_dataset, batch_size=batch_size_per_device, shuffle=True, drop_last=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=False,
        prefetch_factor=prefetch_factor, collate_fn=custom_collate_fn
    ) # sampler=sampler,

    return dataloader