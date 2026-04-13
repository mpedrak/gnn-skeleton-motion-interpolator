import os
import yaml
import torch
import random
import numpy as np
import torch.nn.functional as F

from .rotation import rot_6d_to_quat_torch, quat_to_rot_3x3


def load_configs(filenames, config_dir="./configs/", config_suffix=".yaml"):
    configs = []
    for filename in filenames:
        config_path = config_dir + filename + config_suffix
        if not os.path.isfile(config_path): raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        configs.append(config)

    return configs


def log_string(text, log_path):
    print(text)
    with open(log_path, "a") as log_file:
        log_file.write(text + "\n")


def compute_lerp_batch(start, end, count_to_generate):
    # Start: [B, X], End: [B, X], count_to_generate: int -> Lerp: [B, count_to_generate, X]

    t = torch.linspace(0, 1, steps=count_to_generate + 2, device=start.device)[1 : -1]
    t = t.view(1, count_to_generate, 1)
    start = start.unsqueeze(1)
    end = end.unsqueeze(1)
    result = start + t * (end - start)

    return result


def compute_lerp(start, end, count_to_generate):
    # Start: [X], End: [X], count_to_generate: int -> Lerp: [count_to_generate, X]

    t = torch.linspace(0, 1, steps=count_to_generate + 2, device=start.device)[1 : -1]
    t = t.view(count_to_generate, 1)
    start = start.unsqueeze(0)
    end = end.unsqueeze(0)
    result = start + t * (end - start)

    return result


def set_global_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_worker_seed(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_slerp_batch(start_6d, end_6d, count_to_generate):
    # Start: [B, J, 6] or [B, J*6], End: [B, J, 6] or [B, J*6], count_to_generate: int 
    # -> Slerp (3 x 3 matrix): [B, count_to_generate, J, 3, 3]

    q0 = rot_6d_to_quat_torch(start_6d) 
    q1 = rot_6d_to_quat_torch(end_6d) 

    t = torch.linspace(0, 1, steps=count_to_generate + 2, device=start_6d.device)[1 : -1]
    t = t.view(1, count_to_generate, 1, 1)
    
    q0 = q0.unsqueeze(1)
    q1 = q1.unsqueeze(1)

    cos_half_theta = (q0 * q1).sum(dim=-1, keepdim=True)
    flip_mask = cos_half_theta < 0
    q1 = torch.where(flip_mask, -q1, q1)
    cos_half_theta = torch.abs(cos_half_theta)

    mask = cos_half_theta > 0.9995
    half_theta = torch.acos(torch.clamp(cos_half_theta, -1.0, 1.0))
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta**2) + 1e-8 

    ratio_a = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratio_b = torch.sin(t * half_theta) / sin_half_theta

    res_slerp = ratio_a * q0 + ratio_b * q1
    res_lerp = q0 + t * (q1 - q0)
    res_lerp = F.normalize(res_lerp, dim=-1)

    q_interpolated = torch.where(mask, res_lerp, res_slerp)

    rot_3x3 = quat_to_rot_3x3(q_interpolated) 

    return rot_3x3


def compute_slerp(start_6d, end_6d, count_to_generate):
    # Start: [J, 6] or [J*6], End: [J, 6] or [J*6], count_to_generate: int 
    # -> Slerp (3 x 3 matrix): [count_to_generate, J, 3, 3]
    
    start_6d_batched = start_6d.unsqueeze(0)
    end_6d_batched = end_6d.unsqueeze(0)
    result = compute_slerp_batch(start_6d_batched, end_6d_batched, count_to_generate)
    result = result.squeeze(0)

    return result
