import torch
import yaml
import os
import argparse
import numpy as np

from torch_geometric.loader import DataLoader 
from tqdm import tqdm

from src.dataset import GraphSkeletonDataset
from src.model import SkeletalMotionInterpolator
from src.utils.metrics import (geodesic_rotation_loss, calculate_l2p, calculate_l2q, calculate_npss, 
        compute_smoothness_loss, foot_contact_loss) 
from src.utils.bvh import forward_kinematics_positions_batch, compute_foot_contact, foot_skating_loss
from src.utils.rotation import rot_6d_to_euler_zyx, euler_zyx_to_rot_6d_but_correct_this_time


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


# Evaluate function
def run_benchmark(model, loader, J, F_target, offsets, parent_indices, root_mean, root_std, foot_joint_indices, 
                foot_height_eps, foot_velocity_eps):
    
    n_samples = len(loader.dataset)
    if n_samples == 0 or J == 0 or F_target == 0: raise ValueError("Dataset is empty or has invalid dimensions")
        
    l1 = torch.nn.L1Loss(reduction='sum')
    l2 = torch.nn.MSELoss(reduction='sum')
    rot_geo_1 = lambda pred, target: geodesic_rotation_loss(pred, target, reduction='sum')
    rot_geo_2 = lambda pred, target: (geodesic_rotation_loss(pred, target, reduction='none') ** 2).sum()

    rot_geo_1_sum = 0.0
    rot_geo_2_sum = 0.0
    root_pos_l1_sum = 0.0
    root_pos_l2_sum = 0.0
    all_pos_l1_sum = 0.0
    all_pos_l2_sum = 0.0
    l2p_sum = 0.0
    l2q_sum = 0.0
    npss_sum = 0.0
    
    sm_1_sum = 0.0
    sm_2_sum = 0.0
    sm_3_sum = 0.0

    foot_contact_sum = 0.0
    foot_skating_weighted_motion_sum = 0.0
    foot_skating_num_active_sum = 0.0

    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Test", leave=False):
            
            batch = batch.to(device)
            out = model(batch)

            # Rotations
            rot_pred = out["rot"] # [B * J, F_target * 6]
            rot_tgt = batch.y # [B * J, F_target * 6]

            # All models to this time were trained with wrong 6D representation, 
            # but Euler angles were correct after converting back,
            # fix for training and prediction will be in next version

            rot_pred = rot_pred.view(batch.num_graphs, J, F_target, 6).permute(0, 2, 1, 3) # 6D are wrong
            rot_tgt = rot_tgt.view(batch.num_graphs, J, F_target, 6).permute(0, 2, 1, 3)
            
            rot_euler_pred = rot_6d_to_euler_zyx(rot_pred) # eulers are correct
            rot_tgt_euler = rot_6d_to_euler_zyx(rot_tgt)

            rot_pred = euler_zyx_to_rot_6d_but_correct_this_time(rot_euler_pred) # those 6D are correct
            rot_tgt = euler_zyx_to_rot_6d_but_correct_this_time(rot_tgt_euler)

            rot_pred = torch.tensor(rot_pred, dtype=torch.float32).to(device)
            rot_tgt = torch.tensor(rot_tgt, dtype=torch.float32).to(device)

            rot_pred_flat = rot_pred.permute(0, 2, 1, 3).reshape(batch.num_graphs * J, F_target * 6)
            rot_tgt_flat = rot_tgt.permute(0, 2, 1, 3).reshape(batch.num_graphs * J, F_target * 6)

            rot_geo_1_sum += rot_geo_1(rot_pred_flat, rot_tgt_flat).item()
            rot_geo_2_sum += rot_geo_2(rot_pred_flat, rot_tgt_flat).item()

            # L2Q
            l2q_sum += calculate_l2q(rot_pred_flat, rot_tgt_flat, reduction='sum').item()
            
            # Forward kinematics
            root_pos_pred = out['root_pos']
            root_pos_pred = root_pos_pred.view(batch.num_graphs, F_target, 3)
            root_pos_pred = root_pos_pred * root_std + root_mean

            last_root_pos_absolute = batch.last_root_pos_absolute.view(batch.num_graphs, 1, 3)
            root_pos_pred_absolute = last_root_pos_absolute + torch.cumsum(root_pos_pred, dim=1)

            fk_pos_pred = forward_kinematics_positions_batch( 
                offsets=offsets,
                parent_indices=parent_indices,
                root_pos=root_pos_pred_absolute,
                rot_6d=rot_pred
            ) # [B, F_target, J, 3]

            # Root positions
            fk_pos_tgt_reshaped = batch.fk_pos.view(batch.num_graphs, F_target, J, 3)
            root_pos_tgt_absolute = fk_pos_tgt_reshaped[:, :, 0, :] # [B, F_target, 3], root pos are correct
            root_pos_l1_sum += l1(root_pos_pred_absolute, root_pos_tgt_absolute).item()
            root_pos_l2_sum += l2(root_pos_pred_absolute, root_pos_tgt_absolute).item()

            # All positions
            fk_pos_tgt = forward_kinematics_positions_batch( 
                offsets=offsets,
                parent_indices=parent_indices,
                root_pos=root_pos_tgt_absolute,
                rot_6d=rot_tgt
            ) # [B, F_target, J, 3]

            fk_pos_tgt_flat = fk_pos_tgt.view(batch.num_graphs, -1)
            fk_pos_pred_flat = fk_pos_pred.view(batch.num_graphs, -1)
            all_pos_l1_sum += l1(fk_pos_pred_flat, fk_pos_tgt_flat).item()
            all_pos_l2_sum += l2(fk_pos_pred_flat, fk_pos_tgt_flat).item()

            # L2P
            l2p_sum += calculate_l2p(fk_pos_pred, fk_pos_tgt, reduction='sum').item()

            # NPSS
            fk_pos_pred_npss = fk_pos_pred.permute(0, 2, 3, 1).reshape(batch.num_graphs, J * 3, F_target)
            fk_pos_tgt_npss = fk_pos_tgt_reshaped.permute(0, 2, 3, 1).reshape(batch.num_graphs, J * 3, F_target)
            npss_sum += calculate_npss(fk_pos_pred_npss, fk_pos_tgt_npss, reduction='sum').item()

            # Smoothness
            sm_1_sum += compute_smoothness_loss(fk_pos_pred, fk_pos_tgt_reshaped, order=1, reduction='sum').item()
            sm_2_sum += compute_smoothness_loss(fk_pos_pred, fk_pos_tgt_reshaped, order=2, reduction='sum').item()
            sm_3_sum += compute_smoothness_loss(fk_pos_pred, fk_pos_tgt_reshaped, order=3, reduction='sum').item()

            # Foot contact
            foot_contact_tgt = compute_foot_contact(
                fk_pos=fk_pos_tgt,
                foot_joint_indices=foot_joint_indices,
                contact_height_eps=foot_height_eps,
                contact_velocity_eps=foot_velocity_eps
            )

            foot_contact_pred = compute_foot_contact(
                fk_pos=fk_pos_pred,
                foot_joint_indices=foot_joint_indices,
                contact_height_eps=foot_height_eps,
                contact_velocity_eps=foot_velocity_eps
            )

            foot_contact_sum += foot_contact_loss(foot_contact_tgt, foot_contact_pred, reduction='sum').item()

            # Foot skating
            a, e = foot_skating_loss(
                fk_pos_pred=fk_pos_pred,
                tgt_foot_contact=foot_contact_tgt,
                foot_joint_indices=foot_joint_indices,
                return_elements=True
            )

            foot_skating_weighted_motion_sum += a.item()
            foot_skating_num_active_sum += e.item()


    geo_rot_1 = rot_geo_1_sum / (n_samples * J * F_target)
    geo_rot_1_deg = geo_rot_1 * (180.0 / np.pi)

    geo_rot_2 = np.sqrt(rot_geo_2_sum / (n_samples * J * F_target))
    geo_rot_2_deg = geo_rot_2 * (180.0 / np.pi)

    root_pos_mae = root_pos_l1_sum / (n_samples * F_target * 3)
    
    root_pos_rmse = np.sqrt(root_pos_l2_sum / (n_samples * F_target * 3))

    all_pos_mae = all_pos_l1_sum / (n_samples * F_target * J * 3)
   
    all_pos_rmse = np.sqrt(all_pos_l2_sum / (n_samples * F_target * J * 3))

    l2p_value = l2p_sum / (n_samples * F_target * J)

    l2q_value = l2q_sum / (n_samples * F_target * J)

    npss_value = npss_sum / (n_samples * J * 3)

    sm_1_value = sm_1_sum / (n_samples * (F_target - 1) * J * 3)
    sm_2_value = sm_2_sum / (n_samples * (F_target - 2) * J * 3)
    sm_3_value = sm_3_sum / (n_samples * (F_target - 3) * J * 3)

    foot_contact_value = (foot_contact_sum / (n_samples * F_target * len(foot_joint_indices))) * 100.0
    foot_skating_value = foot_skating_weighted_motion_sum / foot_skating_num_active_sum if foot_skating_num_active_sum > 0 else 0.0

    length_unit = constants["length_unit"]

    return {
        "Per joint" : {
            "L2P value": l2p_value,
            "L2Q value": l2q_value,
            "Rotation MAE [deg]": geo_rot_1_deg,
            "Rotation RMSE [deg]": geo_rot_2_deg,
        },
        "Per joint channel": {
            "NPSS value": npss_value,
            f"Root position MAE [{length_unit}]": root_pos_mae,
            f"Root position RMSE [{length_unit}]": root_pos_rmse,
            f"All positions MAE [{length_unit}]": all_pos_mae,
            f"All positions RMSE [{length_unit}]": all_pos_rmse,
            f"Velocity loss [{length_unit}/frame]": sm_1_value,
            f"Acceleration loss [{length_unit}/frame^2]": sm_2_value,
            f"Jerk loss [{length_unit}/frame^3]": sm_3_value    
        },
        "Per foot joint": {
            "Foot contact error rate [%]": foot_contact_value,
            "Foot skating loss": foot_skating_value
        }
    }


# Dataset
print("Loading dataset")
test_dataset = GraphSkeletonDataset(
    root_dir=constants["benchmark_data_dir"],
    context_len_pre=config["context_len_pre"],
    context_len_post=config["context_len_post"],
    target_len=config["target_len"],
    step=config["step"],
    foot_joint_names=constants["benchmark_foot_joint_names"],
    foot_height_eps=constants["benchmark_foot_height_eps"],
    foot_velocity_eps=constants["benchmark_foot_velocity_eps"]
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

model_path = constants["model_path"] + filename + constants["model_suffix"]
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
print(f"Loaded checkpoint: {model_path}")

# Testing
print("Starting benchmark on test set")
root_stats_path = constants["root_stats_path"] + filename + constants["root_stats_suffix"]
stats = np.load(root_stats_path)
print(f"Loaded root stats from: {root_stats_path}")
root_mean = torch.tensor(stats["mean"], dtype=torch.float32).to(device).view(1, 1, 3)
root_std = torch.tensor(stats["std"], dtype=torch.float32).to(device).view(1, 1, 3)

results = run_benchmark(
    model=model,
    loader=test_loader,
    J=test_dataset.num_joints,
    F_target=config["target_len"],
    offsets=test_dataset.offsets.to(device),
    parent_indices=test_dataset.parent_indices.to(device),
    root_mean=root_mean,
    root_std=root_std,
    foot_joint_indices=test_dataset.foot_joint_indices.to(device),
    foot_height_eps=constants["benchmark_foot_height_eps"],
    foot_velocity_eps=constants["benchmark_foot_velocity_eps"]
)

os.makedirs(constants["benchmark_log_path"], exist_ok=True)
test_log_path = constants["benchmark_log_path"] + filename + constants["log_suffix"]
if os.path.exists(test_log_path):
    os.remove(test_log_path)

def log_str(str):
    print(str)
    with open(test_log_path, "a") as log_file:
        log_file.write(str + "\n")

log_str("\n--- Benchmark Results ---")
for metric_group, metric_values in results.items():
    log_str(f"\n{metric_group}:")
    for metric_name, value in metric_values.items():
        log_str(f"  {metric_name}: {' ' * (32 - len(metric_name))} {value:.7f}")

print()
