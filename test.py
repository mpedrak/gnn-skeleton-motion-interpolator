import torch
import yaml
import os
import argparse

from torch_geometric.loader import DataLoader 
from tqdm import tqdm

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


def evaluate(model, loader, root_loss_weight, J, F_target, node_features):
    model.eval()
    mse = torch.nn.MSELoss()
    total_loss = 0.0
    num_samples = 0
    mse_root_total = 0.0
    mse_rot_total = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Test", leave=False):
            batch = batch.to(device)
            out = model(batch)

            # rot_pred = out["rot"].view(batch.num_graphs * J, F_target * node_features)
            rot_pred = out["rot"]
            loss_rot = geodesic_rotation_loss(rot_pred, batch.y)
            
            root_tgt = batch.root_tgt_norm.view(batch.num_graphs, -1) 
            loss_root = mse(out['root_norm'], root_tgt)
            loss = loss_rot + root_loss_weight * loss_root

            total_loss += loss.item() * batch.num_graphs
            num_samples += batch.num_graphs
            mse_rot_total += loss_rot.item()
            mse_root_total += loss_root.item()
            n_batches += 1

    avg_loss = total_loss / max(1, num_samples)
    avg_mse_root = mse_root_total / max(1, n_batches)  
    avg_mse_rot = mse_rot_total / max(1, n_batches)

    return {
        "overall_mse": avg_loss,
        "root_mse_norm": avg_mse_root,
        "rot_6d_mse": avg_mse_rot
    }


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
    graph_features=config["graph_features"],
    num_joints=test_dataset.num_joints
)
model = model.to(device)

model_path = config["model_path"]
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
print(f"Loaded checkpoint: {model_path}")

print("Starting evaluation on test set")
J = test_dataset.num_joints
F_target = config["target_len"]
node_features = config["node_features"]
results = evaluate(model, test_loader, config["root_loss_weight"], J, F_target, node_features)

test_log_path = config["test_log_path"]

if os.path.exists(test_log_path):
    os.remove(test_log_path)

os.makedirs("results", exist_ok=True)

def log_str(str):
    print(str)
    with open(test_log_path, "a") as log_file:
        log_file.write(str + "\n")

log_str("\n--- Test Results ---")
log_str(f"Geo Loss 6D rotations:                     {results['rot_6d_mse']:.7f}")
log_str(f"MSE root positions (normalized deltas):    {results['root_mse_norm']:.7f}")
log_str(f"Loss sum (with root loss weight):          {results['overall_mse']:.7f}")
