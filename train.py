import torch
import yaml
import numpy as np
import os
import argparse

from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.dataset import GraphSkeletonDataset
from src.model import SkeletalMotionInterpolator
from src.utils.rotation import geodesic_rotation_loss
from src.utils.bvh import forward_kinematics_positions_batch


# Config
config_dir = "./config/"

parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
args = parser.parse_args()

filename = args.config
config_path = config_dir + filename + ".yaml"
if not os.path.isfile(config_path):
    raise FileNotFoundError(f"Config file not found: {config_path}")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

constants_path = config_dir + "constants.yaml"
if not os.path.isfile(constants_path):
    raise FileNotFoundError(f"Constants file not found: {constants_path}")

with open(constants_path, "r") as f:
    constants = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Dataset
print("Loading dataset")
dataset = GraphSkeletonDataset(
    root_dir=config["train_data_dir"],
    context_len_pre=config["context_len_pre"],
    context_len_post=config["context_len_post"],
    target_len=config["target_len"],
    step=config["step"]
)
print(f"Dataset ready with {len(dataset)} samples")

os.makedirs(constants["root_stats_path"], exist_ok=True)
root_stats_path = constants["root_stats_path"] + filename + constants["root_stats_suffix"]
np.savez(root_stats_path, mean=dataset.root_mean.numpy(), std=dataset.root_std.numpy())
print(f"Saved root delta stats in: {root_stats_path}")

n_total = len(dataset)
n_val = max(1, int(n_total * config["validation_split"]))
n_train = n_total - n_val
train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(7))

batch_size = config["batch_size"]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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

# Training 
start_lr = config["start_lr"]
optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=config["lr_factor"],     
    patience=config["lr_patience"],     
    min_lr=config["min_lr"]
)

mse = torch.nn.MSELoss()
mae = torch.nn.L1Loss()

os.makedirs(constants["train_log_path"], exist_ok=True)
train_log_path = constants["train_log_path"] + filename + constants["log_suffix"]
if os.path.exists(train_log_path):
    os.remove(train_log_path)

def log_str(str):
    print(str)
    with open(train_log_path, "a") as log_file:
        log_file.write(str + "\n")

os.makedirs(constants["model_path"], exist_ok=True)
model_path = constants["model_path"] + filename + constants["model_suffix"]
root_loss_weight = config["root_loss_weight"]
fk_loss_weight = config["fk_loss_weight"]
patience = config["patience"]
epochs = config["epochs"]
F_target = config["target_len"]
node_features = config["node_features"]

J = dataset.num_joints
root_mean = dataset.root_mean.to(device).view(1, 1, 3)
root_std = dataset.root_std.to(device).view(1, 1, 3)
parent_indices = dataset.parent_indices.to(device)
offsets = dataset.offsets.to(device)

best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(1, epochs + 1):
    log_str(f"\n--- Epoch {epoch}/{epochs} ---")

    current_lr = optimizer.param_groups[0]["lr"]
    log_str(f"Learning rate:                {current_lr:.3e}")
   
    model.train()
    total_train_loss = 0.0

    for batch in tqdm(train_loader, desc="Train", leave=False):
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch)
        
        # Rotations
        rot_pred = out["rot"]
        loss_rot = geodesic_rotation_loss(rot_pred, batch.y)
        
        # Root positions
        root_pos_tgt = batch.root_pos_tgt.view(batch.num_graphs, -1) 
        root_pos_pred = out['root_pos']
        loss_root_pos = mse(root_pos_pred, root_pos_tgt)

        # Forward kinematics 
        rot_pred = rot_pred.view(batch.num_graphs, J, F_target, 6).permute(0, 2, 1, 3) # [B * J, F_target * 6] -> [B, F_target, J, 6]
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

        # Total loss
        loss = loss_rot + root_loss_weight * loss_root_pos + fk_loss_weight * loss_fk
        loss.backward()

        optimizer.step()
        total_train_loss += loss.item() * batch.num_graphs

    avg_train_loss = total_train_loss / len(train_dataset)
    log_str(f"Training loss:                {avg_train_loss:.7f}")

    model.eval()
    total_val_loss = 0.0
    total_rot_loss = 0.0
    total_root_loss = 0.0
    total_fk_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val", leave=False):
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
            rot_pred = rot_pred.view(batch.num_graphs, J, F_target, 6).permute(0, 2, 1, 3) # [B * J, F_target * 6] -> [B, F_target, J, 6]
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
            loss = loss_rot + root_loss_weight * loss_root_pos + fk_loss_weight * loss_fk
            total_val_loss += loss.item() * batch.num_graphs
            total_rot_loss += loss_rot.item() * batch.num_graphs
            total_root_loss += loss_root_pos.item() * batch.num_graphs
            total_fk_loss += loss_fk.item() * batch.num_graphs

    avg_val_loss = total_val_loss / len(val_dataset)
    log_str(f"Validation loss:              {avg_val_loss:.7f}")
    log_str(f"Rotations geodesic loss:      {total_rot_loss / len(val_dataset):.7f}")
    log_str(f"Root positions MSE:           {total_root_loss / len(val_dataset):.7f}")
    log_str(f"FK positions MAE:             {total_fk_loss / len(val_dataset):.7f}")

    scheduler.step(avg_val_loss)
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        log_str("Validation loss improved, saving checkpoint")
        torch.save(model.state_dict(), model_path)
        log_str(f"Model saved to: {model_path}")
    else:
        epochs_no_improve += 1
        log_str(f"No improvement in validation loss for {epochs_no_improve} epochs")
        if epochs_no_improve >= patience:
            log_str(f"Early stopped at epoch {epoch}")
            break

log_str("Training complete")
