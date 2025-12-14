import torch
import yaml
import os
import argparse
import numpy as np

from torch_geometric.loader import DataLoader 
from tqdm import tqdm

from src.dataset import GraphSkeletonDataset
from src.model import SkeletalMotionInterpolator
from src.utils.rotation import geodesic_rotation_loss
from src.utils.bvh import forward_kinematics_positions_batch


# Config
config_dir = "./config/"

parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
args = parser.parse_args()

config_path = config_dir + args.config + ".yaml"
if not os.path.isfile(config_path):
    raise FileNotFoundError(f"Config file not found: {config_path}")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Evaluate function
def evaluate(model, loader, root_loss_weight, fk_loss_weight, J, F_target, offsets, parent_indices, root_mean, root_std):
    model.eval()
    mse = torch.nn.MSELoss()
    mae = torch.nn.L1Loss()
    total_loss = 0.0
    num_samples = 0
    mse_root_total = 0.0
    geo_rot_total = 0.0
    mae_fk_total = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Test", leave=False):
            batch = batch.to(device)

            out = model(batch)

            # Rotations
            rot_pred = out["rot"]
            loss_rot = geodesic_rotation_loss(rot_pred, batch.y)
            
            # Root positions
            root_pos_tgt = batch.root_pos_tgt.view(batch.num_graphs, -1) 
            root_pos_pred = out['root_pos']
            loss_root_pos = mse(root_pos_pred, root_pos_tgt)

            # Forward kinematics 
            rot_pred = rot_pred.view(batch.num_graphs, J, F_target, 6).permute(0, 2, 1, 3) 
            root_pos_pred = root_pos_pred.view(batch.num_graphs, F_target, 3)
            root_pos_pred = root_pos_pred * root_std + root_mean

            fk_pos_pred = forward_kinematics_positions_batch(
                offsets=offsets,
                parent_indices=parent_indices,
                root_pos=root_pos_pred,
                rot_6d=rot_pred
            ) 

            fk_pos_tgt = batch.fk_pos.view(batch.num_graphs, -1)
            fk_pos_pred = fk_pos_pred.view(batch.num_graphs, -1)
            loss_fk = mae(fk_pos_pred, fk_pos_tgt)

            # Losses
            loss = fk_loss_weight * loss_fk + root_loss_weight * loss_root_pos + loss_rot
            total_loss += loss.item() * batch.num_graphs
            num_samples += batch.num_graphs

            geo_rot_total += loss_rot.item()
            mse_root_total += loss_root_pos.item()
            mae_fk_total += loss_fk.item()
            n_batches += 1

    avg_loss = total_loss / max(1, num_samples)
    avg_mse_root = mse_root_total / max(1, n_batches)  
    avg_geo_rot = geo_rot_total / max(1, n_batches)
    avg_mae_fk = mae_fk_total / max(1, n_batches)

    return {
        "total": avg_loss,
        "mse_root": avg_mse_root,
        "geo_rot": avg_geo_rot,
        "mae_fk": avg_mae_fk
    }


# Dataset
print("Loading dataset")
test_dataset = GraphSkeletonDataset(
    root_dir=config["test_data_dir"],
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
    hidden_dim=config["hidden_dim"],
    hidden_layers=config["hidden_layers"],
    root_pos_hidden_dim=config["root_pos_hidden_dim"],
    heads=config["heads"],
    dropout=config["dropout"],
    node_features=config["node_features"],
    graph_features=config["graph_features"]
)
model = model.to(device)

model_path = config["model_path"]
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
print(f"Loaded checkpoint: {model_path}")

# Testing
print("Starting evaluation on test set")
stats = np.load(config["root_stats_path"])
root_mean = torch.tensor(stats["mean"], dtype=torch.float32).to(device).view(1, 1, 3)
root_std = torch.tensor(stats["std"], dtype=torch.float32).to(device).view(1, 1, 3)

results = evaluate(
    model=model,
    loader=test_loader,
    root_loss_weight=config["root_loss_weight"],
    fk_loss_weight=config["fk_loss_weight"],
    J=test_dataset.num_joints,
    F_target=config["target_len"],
    offsets=test_dataset.offsets.to(device),
    parent_indices=test_dataset.parent_indices.to(device),
    root_mean=root_mean,
    root_std=root_std
)

test_log_path = config["test_log_path"]
os.makedirs("results", exist_ok=True)
if os.path.exists(test_log_path):
    os.remove(test_log_path)

def log_str(str):
    print(str)
    with open(test_log_path, "a") as log_file:
        log_file.write(str + "\n")

log_str("\n--- Test Results ---")
log_str(f"Total loss - weighted sum:       {results['total']:.7f}")
log_str(f"6D rotations geodesic loss:      {results['geo_rot']:.7f}")
log_str(f"Root positions MSE:              {results['mse_root']:.7f}")
log_str(f"FK positions MAE:                {results['mae_fk']:.7f}")
