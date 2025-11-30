import torch
import yaml
import numpy as np
import os
import argparse

from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch.utils.data import random_split

from src.dataset import GraphSkeletonDataset
from src.model import SkeletalMotionInterpolator
from src.utils.rotation import geodesic_rotation_loss


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

print("Loading dataset")
dataset = GraphSkeletonDataset(
    root_dir=config["train_data_dir"],
    context_len_pre=config["context_len_pre"],
    context_len_post=config["context_len_post"],
    target_len=config["target_len"],
    step=config["step"]
)
print(f"Dataset ready with {len(dataset)} samples")

root_stats_path = config["root_stats_path"]
os.makedirs("checkpoints", exist_ok=True)
np.savez(root_stats_path, mean=dataset.root_mean.numpy(), std=dataset.root_std.numpy())
print(f"Saved root delta stats in: {root_stats_path}")

n_total = len(dataset)
n_val = max(1, int(n_total * config["validation_split"]))
n_train = n_total - n_val
train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(7))

batch_size = config["batch_size"]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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

optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
mse = torch.nn.MSELoss()

best_val_loss = float('inf')
epochs_no_improve = 0

model_path = config["model_path"]
root_loss_weight = config["root_loss_weight"]
patience = config["patience"]
epochs = config["epochs"]
train_log_path = config["train_log_path"]

if os.path.exists(train_log_path):
    os.remove(train_log_path)

os.makedirs("results", exist_ok=True)

def log_str(str):
    print(str)
    with open(train_log_path, "a") as log_file:
        log_file.write(str + "\n")
   
for epoch in range(1, epochs + 1):
    log_str(f"\n--- Epoch {epoch}/{epochs} ---")
   
    model.train()
    total_train_loss = 0.0

    for batch in tqdm(train_loader, desc="Train", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        # loss_rot = mse(out['rot'], batch.y)
        loss_rot = geodesic_rotation_loss(out['rot'], batch.y)
        root_tgt = batch.root_tgt_norm.view(batch.num_graphs, -1) 
        loss_root = mse(out['root_norm'], root_tgt)
        loss = loss_rot + root_loss_weight * loss_root
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * batch.num_graphs

    avg_train_loss = total_train_loss / len(train_dataset)
    log_str(f"Train loss:                         {avg_train_loss:.7f}")

    model.eval()
    total_val_loss = 0.0
    total_rot_loss = 0.0
    total_root_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val", leave=False):
            batch = batch.to(device)
            out = model(batch)
            # loss_rot = mse(out['rot'], batch.y)
            loss_rot = geodesic_rotation_loss(out['rot'], batch.y)
            root_tgt = batch.root_tgt_norm.view(batch.num_graphs, -1) 
            loss_root = mse(out['root_norm'], root_tgt)
            loss = loss_rot + root_loss_weight * loss_root
            total_val_loss += loss.item() * batch.num_graphs
            total_rot_loss += loss_rot.item() * batch.num_graphs
            total_root_loss += loss_root.item() * batch.num_graphs

    avg_val_loss = total_val_loss / len(val_dataset)
    log_str(f"Validation loss:                    {avg_val_loss:.7f}")
    # log_str(f"Validation MSE 6D rotations:      {total_rot_loss / len(val_dataset):.7f}")
    log_str(f"Validation Geo Loss 6D rotations:   {total_rot_loss / len(val_dataset):.7f}")
    log_str(f"Validation MSE root positions:      {total_root_loss / len(val_dataset):.7f}")
    
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
