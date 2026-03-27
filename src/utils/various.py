import os
import yaml
import torch


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
