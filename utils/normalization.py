import math
import torch

def rotation_norm(end_effector_state):
    '''
    Converts Euler angles [r, p, y] in the end-effector state to their sine and cosine representations.

    Args:
        end_effector_state (list): A list like [x, y, z, r, p, y, ...], where r, p, y are roll, pitch, yaw angles.

    Returns:
        list: [x, y, z, sin(r), cos(r), sin(p), cos(p), sin(y), cos(y), ...]
    '''
    r, p, y = end_effector_state[3:6]
    values = [math.sin(r), math.cos(r), math.sin(p), math.cos(p), math.sin(y), math.cos(y)]
    return end_effector_state[:3] + values + end_effector_state[6:]

def rotation_denorm(norm_end_effector_state):
    '''
    Recovers Euler angles [r, p, y] from their sine and cosine representations in the normalized end-effector state.

    Args:
        norm_end_effector_state (list): A list like [x, y, z, sin(r), cos(r), sin(p), cos(p), sin(y), cos(y), ...]

    Returns:
        list: [x, y, z, r, p, y, ...], where r, p, y are recovered roll, pitch, yaw angles.
    '''
    x, y, z = norm_end_effector_state[:3]
    sin_r, cos_r = norm_end_effector_state[3], norm_end_effector_state[4]
    sin_p, cos_p = norm_end_effector_state[5], norm_end_effector_state[6]
    sin_y, cos_y = norm_end_effector_state[7], norm_end_effector_state[8]

    r = math.atan2(sin_r, cos_r)
    p = math.atan2(sin_p, cos_p)
    y_ = math.atan2(sin_y, cos_y)

    end_effector_state = [x, y, z, r, p, y_] + norm_end_effector_state[9:]
    return end_effector_state

def min_max_norm(values, key_stats, eps=1e-8):
    """
    Per-dimension min-max normalization to [-1, 1].

    Args:
        values (array-like): shape (N, L) or (L,), can be list, numpy array, or torch tensor
        key_stats (dict): A dict, where key_stats["min"] and key_stats["max"] are 1D numpy.ndarray shape (L,)
        eps (float): Epsilon for division safety.

    Returns:
        torch.Tensor: normalized tensor shape (N, L) or [1, L], values in [-1, 1]
    """
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values)
    
    if values.ndim == 1:
        values = values.unsqueeze(0)
    
    min_values = torch.from_numpy(key_stats["min"]).to(dtype=values.dtype, device=values.device)
    max_values = torch.from_numpy(key_stats["max"]).to(dtype=values.dtype, device=values.device)
    normed_0_1 = (values - min_values) / (max_values - min_values + eps)
    normed_minus1_1 = normed_0_1 * 2 - 1
    return normed_minus1_1

def min_max_denorm(values, key_stats, eps=1e-8):
    """
    Inverse of min-max normalization to [-1, 1].

    Args:
        values (array-like): shape (N, L) or (L,), can be list, numpy array, or torch tensor. Normalized values in [-1, 1]
        key_stats (dict): key_stats["min"], key_stats["max"] 1D numpy.ndarray, shape (L,)
        eps (float): Epsilon added during normalization

    Returns:
        torch.Tensor: shape (N, L), denormalized values
    """
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values)
    
    if values.ndim == 1:
        values = values.unsqueeze(0)
    
    min_values = torch.from_numpy(key_stats["min"]).to(dtype=values.dtype, device=values.device)
    max_values = torch.from_numpy(key_stats["max"]).to(dtype=values.dtype, device=values.device)
    scale = (max_values - min_values + eps)

    # Inverse transform: x = ((normed + 1)/2) * scale + min
    orig_values = ((values + 1) / 2) * scale + min_values
    return orig_values

def xarm_tabletop_post_action_denorm(action_chunk, dataset_meta):
    # denormalize each dim from [-1, 1] to [min, max]
    action_chunk = min_max_denorm(action_chunk, dataset_meta.stats["action"])
    action_chunk = action_chunk.tolist()
    # denormalize the rotation dims, i.e., [x, y, z, sin(r), cos(r), ...] -> [x, y, z, r, ...]
    action_chunk = [rotation_denorm(action) for action in action_chunk]
    # the gripper's state is 0 or 1
    for action in action_chunk:
        action[-1] = 1 if action[-1] > 0.5 else 0

    # # Determine the number of trusted actions based on changes in the gripper's state,
    # # and we return the first few trusted actions
    # fixed_trust_action_num = 8
    # if len(action_chunk) > fixed_trust_action_num:
    #     trust_action_chunk = action_chunk[:fixed_trust_action_num]
    #     initial_grapper_state = action_chunk[fixed_trust_action_num-1][-1] # 0: open, 1: closed
    #     for action in action_chunk[fixed_trust_action_num:]:
    #         if initial_grapper_state != action[-1]:
    #             break
    #         else:
    #             trust_action_chunk.append(action)
    #     return trust_action_chunk
    # else:
    #     return action_chunk

    return action_chunk

def libero_post_action_denorm(action_chunk, dataset_meta):
    # denormalize each dim from [-1, 1] to [min, max]
    action_chunk = min_max_denorm(action_chunk, dataset_meta.stats["action"])
    action_chunk = action_chunk.tolist()

    return action_chunk
def standard_post_action_denorm(action_chunk, dataset_meta):
    # denormalize each dim from [-1, 1] to [min, max]
    action_chunk = min_max_denorm(action_chunk, dataset_meta.stats["action"])
    action_chunk = action_chunk.tolist()
    return action_chunk