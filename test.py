import torch
import os
import argparse
import numpy as np

from torch_geometric.loader import DataLoader 
from tqdm import tqdm

from src.dataset import GraphSkeletonDataset
from src.model import SkeletalMotionInterpolator
from src.calculate_loss import calculate_loss
from src.utils.various import load_configs, log_string


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
args = parser.parse_args()
filename = args.config

config, constants = load_configs([filename, "constants"])
print(f"Loaded config: {filename}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True


# Evaluate function
def evaluate(model, loader, offsets, parent_indices, root_mean, root_std, loss_weights, n_samples, log_str):
    
    model.eval()

    l1_func = torch.nn.L1Loss()
    l2_func = torch.nn.MSELoss()

    total_loss = 0.0
    total_rot_loss = 0.0
    total_root_pos_loss = 0.0
    total_fk_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Test", leave=False):
            
            batch = batch.to(device)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16): 
                out = model(batch)

            out["rot"] = out["rot"].float()
            out["root_pos"] = out["root_pos"].float()

            loss, rot_geo_loss, root_pos_loss, fk_pos_loss = calculate_loss(
                out=out, 
                batch=batch, 
                l1_func=l1_func, 
                l2_func=l2_func, 
                root_std=root_std, 
                root_mean=root_mean, 
                offsets=offsets, 
                parent_indices=parent_indices, 
                loss_weights=loss_weights
            )

            total_loss += loss.item() * batch.num_graphs
            total_rot_loss += rot_geo_loss.item() * batch.num_graphs
            total_root_pos_loss += root_pos_loss.item() * batch.num_graphs
            total_fk_loss += fk_pos_loss.item() * batch.num_graphs           

        avg_loss = total_loss / n_samples
        avg_rot_loss = total_rot_loss / n_samples
        avg_root_pos_loss = total_root_pos_loss / n_samples
        avg_fk_loss = total_fk_loss / n_samples

        log_str(f"Test loss:                      {avg_loss:.7f}")
        log_str(f"'- Rotations Geodesic L1:       '- {avg_rot_loss:.4f}")
        log_str(f"'- Root positions L2:           '- {avg_root_pos_loss:.4f}")
        log_str(f"'- FK positions L1:             '- {avg_fk_loss:.4f}")  


# Dataset
print("Loading dataset")
test_dataset = GraphSkeletonDataset(
    root_dir=constants["test_data_dir"],
    context_len_pre=config["context_len_pre"],
    context_len_post=config["context_len_post"],
    target_len=config["target_len"],
    step=config["step"]
)
print(f"Dataset ready with {len(test_dataset)} samples")

test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)


# Model
model = SkeletalMotionInterpolator(
    context_len_pre=config["context_len_pre"],
    context_len_post=config["context_len_post"],
    target_len=config["target_len"],
    rot_gnn_params=config["rot_gnn_params"],
    root_pos_mlp_params=config["root_pos_mlp_params"]
)
model = model.to(device)

model_path = constants["models_path"] + filename + constants["models_suffix"]
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
print(f"Loaded checkpoint: {model_path}")


# Logging
os.makedirs(constants["test_log_path"], exist_ok=True)
test_log_path = constants["test_log_path"] + filename + constants["log_suffix"]
if os.path.exists(test_log_path):
    os.remove(test_log_path)

log_str = lambda text: log_string(text=text, log_path=test_log_path)


# Testing
root_stats_path = constants["root_stats_path"] + filename + constants["root_stats_suffix"]
stats = np.load(root_stats_path)
root_mean = torch.tensor(stats["mean"], dtype=torch.float32).to(device).view(1, 1, 3)
root_std = torch.tensor(stats["std"], dtype=torch.float32).to(device).view(1, 1, 3)
print(f"Loaded root stats from: {root_stats_path}")

parent_indices = test_dataset.parent_indices.to(device)
offsets = test_dataset.offsets.to(device)

print("\nStarting evaluation on test set")
evaluate(
    model=model, 
    loader=test_loader, 
    offsets=offsets, 
    parent_indices=parent_indices, 
    root_mean=root_mean, 
    root_std=root_std, 
    loss_weights=config["loss_weights"], 
    n_samples=len(test_dataset), 
    log_str=log_str
)

print("\nTesting completed")
print()
