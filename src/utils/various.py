import os
import yaml
import torch
import random
import numpy as np


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
