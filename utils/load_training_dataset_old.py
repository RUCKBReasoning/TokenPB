from torch.utils.data import Dataset
from collections import OrderedDict
import os
import torch
import random
import time

import torch

def find_subtensor_index(tensor, sub_tensor):
    """
    Find the first occurrence of sub_tensor as a consecutive subsequence in a 1D tensor.
    Returns the start index, or -1 if not found.

    Args:
        tensor (torch.Tensor): 1D tensor to search in.
        sub_tensor (torch.Tensor): 1D tensor to search for.

    Returns:
        int: start index of the first match, or -1 if not found
    """
    n = sub_tensor.numel()
    if n == 0 or n > tensor.numel():
        return -1
    # Create sliding windows of length n on the original tensor
    windows = tensor.unfold(0, n, 1)  # shape: [L-n+1, n]
    # Compare each window with sub_tensor
    matches = (windows == sub_tensor).all(dim=1)
    # Find indices where a full match occurs
    idxs = torch.where(matches)[0]
    # Return the first matching index, or -1 if no match
    return idxs[0].item() if idxs.numel() > 0 else -1

def mask_prompt_loss_func(labels):
    '''
    Mask everything before assistant_start_token_ids with -100
    '''
    # NOTE: the start token ids of the Assistant is [151644, 77091], meaning "<|im_start|>assistant" in Qwen2's tokenizer
    assistant_start_token_ids = torch.tensor([151644, 77091], dtype=labels.dtype)
    assistant_start_idx = find_subtensor_index(labels, assistant_start_token_ids)

    if assistant_start_idx == -1:
        print("we do not find assistant start tokens in this sample.")
        return labels
    else:
        # Set all elements before 'index' to -100 (in-place)
        labels[:assistant_start_idx] = -100
        return labels

class ReasoningVLADataset(Dataset):
    def __init__(self, dataset_path_list: list[str], mask_prompt_loss: bool, dataset_type: str = "for_vla"):
        super().__init__()
        print("dataset_path_list:", dataset_path_list)
        self.mask_prompt_loss = mask_prompt_loss

        if dataset_type == "for_vlm":
            keys = ['attention_mask', 'input_ids', 'pixel_values', 'image_grid_thw', 'labels']
        elif dataset_type == "for_vla":
            keys = ['attention_mask', 'input_ids', 'pixel_values', 'image_grid_thw', 'labels', 
                    'action', 'action_mask', 'embodiment_id', 'observation.state', 'state_mask']
        else:
            raise ValueError("dataset_type should be in [for_vlm, for_vla].")

        chunk_dirs = []
        for dataset_path in dataset_path_list:
            for chunk_id in os.listdir(os.path.join(dataset_path, "train")):
                chunk_dirs.append(os.path.join(dataset_path, "train", chunk_id)) # "preprocessed_data"

        self.chunk_dirs = chunk_dirs
        # Record the number of samples in each chunk
        self.chunk_lens = []
        self.cum_lens = []    # Cumulative lengths, used for global index lookup
        self.chunk_tensor_paths = []  # Record the absolute paths for pt files in each chunk
        # Only read the shape at this step, do not load content
        for chunk_dir in self.chunk_dirs:
            if not os.path.exists(os.path.join(chunk_dir, keys[0]+'.pt')):
                print(f"skip loading {chunk_dir}")
                continue
            tensor_files = {
                name: os.path.join(chunk_dir, name+'.pt')
                for name in keys
            }
            # Use torch.load(mmap=True) to read only the shape
            sample_tensor = torch.load(tensor_files['input_ids'], mmap=True)
            # n is the number of data samples in the chunk
            n = sample_tensor.shape[0]
            self.chunk_lens.append(n)
            self.chunk_tensor_paths.append(tensor_files)
        # Compute total number of samples and chunk offsets
        self.cum_lens = [0]
        for n in self.chunk_lens:
            self.cum_lens.append(self.cum_lens[-1] + n)
        self.total_len = self.cum_lens[-1]
        # File handles opened (use LRU cache)
        self.open_chunks = OrderedDict()
        self.max_open_chunks = 2048

    def __len__(self):
        return self.total_len
    
    def _find_chunk_and_offset(self, idx):
        # Binary search is faster if there are many chunks
        # Here assume the number of chunks is small and keep it simple
        for chunk_idx in range(len(self.chunk_lens)):
            if self.cum_lens[chunk_idx] <= idx < self.cum_lens[chunk_idx+1]:
                offset = idx - self.cum_lens[chunk_idx]
                return chunk_idx, offset
        raise IndexError(f"Index {idx} out of range.")

    def _load_chunk(self, chunk_idx):
        """
        Open all .pt files for the specified chunk using mmap (without fully loading into memory)
        and cache them in self.open_chunks using an LRU policy.
        """
        if chunk_idx in self.open_chunks:
            # Move the accessed chunk to the end to mark it as recently used (LRU)
            self.open_chunks.move_to_end(chunk_idx)
            return  # Chunk already loaded

        # If the cache is full, remove the least recently used chunk (from the start)
        if len(self.open_chunks) >= self.max_open_chunks:
            old_idx, old_handle = self.open_chunks.popitem(last=False)
            del old_handle

        # Load the new chunk and cache it
        tensor_handles = {}
        paths = self.chunk_tensor_paths[chunk_idx]
        for name, path in paths.items():
            tensor_handles[name] = torch.load(path, mmap=True) # mmap=True
        self.open_chunks[chunk_idx] = tensor_handles

    def __getitem__(self, idx):
        # Step 1: Locate which chunk and the corresponding offset within the chunk
        chunk_idx, offset = self._find_chunk_and_offset(idx)
        # Step 2: Load the chunk (lazy loading, each chunk is only loaded once)
        self._load_chunk(chunk_idx)
        chunk = self.open_chunks[chunk_idx]
        # Step 3: For all files, fetch the sample at the same offset
        sample = {
            name: tensor[offset]
            for name, tensor in chunk.items()
        }
        # Step 4: Mask the loss of padding tokens, replace all occurrences of 151643 with -100 
        # (NOTE: 151643 is the pad token id of Qwen2's tokenizer)
        sample["labels"][sample["labels"] == 151643] = -100

        if self.mask_prompt_loss:
            sample["labels"] = mask_prompt_loss_func(sample["labels"])

        return sample

    def close(self):
        # Manual resource release, if needed
        self.open_chunks.clear()

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

if __name__ == "__main__":
    dataset = ReasoningVLADataset(
        ["/pretrain_data/dataset_lerobot_vlm/vqa/BLINK_4", "/pretrain_data/dataset_lerobot_vlm/video/ego4d_goalstep_train_2"],
        True,
        "for_vlm"
    )

    random_idx = random.randint(0, len(dataset))
    sample = dataset[random_idx]

    # from transformers import AutoProcessor
    # processor = AutoProcessor.from_pretrained("/root/hf_models/Qwen/Qwen2.5-VL-3B-Instruct")

    # for token_index, (input_id, label) in enumerate(zip(sample["input_ids"], sample["labels"])):
    #     input_id = input_id.item()
    #     label = label.item()
    #     print(f"{token_index}: {input_id} | {repr(processor.tokenizer.decode(input_id))} -> {label}")

    # for key, data in dataset[random_idx].items():
    #     print(key)
    #     print(data.shape)
    #     print(data)