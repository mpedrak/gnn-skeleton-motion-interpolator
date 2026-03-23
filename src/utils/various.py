import os
import yaml


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
