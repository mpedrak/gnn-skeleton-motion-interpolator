import torch
import os
import argparse
import numpy as np

from torch_geometric.loader import DataLoader 
from tqdm import tqdm

from src.dataset import GraphSkeletonDataset
from src.model import SkeletalMotionInterpolator
from src.utils.metrics import geodesic_rotation_loss, calculate_l2p, calculate_l2q, calculate_npss
from src.utils.metrics import calculate_smoothness_loss, calculate_foot_contact_loss 
from src.utils.bvh import forward_kinematics_positions_batch, compute_foot_contact, foot_skating_loss, get_joint_indices_by_name
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
def run_benchmark(model, loader, offsets, parent_indices, root_mean, root_std, n_samples, benchmark_foot_params, joint_names):
           
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

    foot_joint_indices = get_joint_indices_by_name(
        all_joint_names=joint_names, 
        target_joint_names=benchmark_foot_params["joint_names"]
    )

    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Test", leave=False):
            
            batch = batch.to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16): 
                out = model(batch)

            out["rot"] = out["rot"].float()
            out["root_pos"] = out["root_pos"].float()

            # Rotations
            rot_pred = out["rot"] 
            rot_geo_1_sum += rot_geo_1(rot_pred, batch.y).item()
            rot_geo_2_sum += rot_geo_2(rot_pred, batch.y).item()

            # L2Q
            l2q_sum += calculate_l2q(rot_pred, batch.y, reduction='sum').item()
            
            BxJ, Fx6 = rot_pred.shape
            F_target = Fx6 // 6
            J = BxJ // batch.num_graphs  
            rot_pred = rot_pred.view(batch.num_graphs, J, F_target, 6).permute(0, 2, 1, 3) 

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
            root_pos_tgt_absolute = fk_pos_tgt_reshaped[:, :, 0, :] # [B, F_target, 3]
            root_pos_l1_sum += l1(root_pos_pred_absolute, root_pos_tgt_absolute).item()
            root_pos_l2_sum += l2(root_pos_pred_absolute, root_pos_tgt_absolute).item()

            # All positions
            fk_pos_tgt_flat = fk_pos_tgt_reshaped.view(batch.num_graphs, -1)
            fk_pos_pred_flat = fk_pos_pred.view(batch.num_graphs, -1)
            all_pos_l1_sum += l1(fk_pos_pred_flat, fk_pos_tgt_flat).item()
            all_pos_l2_sum += l2(fk_pos_pred_flat, fk_pos_tgt_flat).item()

            # L2P
            l2p_sum += calculate_l2p(fk_pos_pred, fk_pos_tgt_reshaped, reduction='sum').item()

            # NPSS
            fk_pos_pred_npss = fk_pos_pred.permute(0, 2, 3, 1).reshape(batch.num_graphs, J * 3, F_target)
            fk_pos_tgt_npss = fk_pos_tgt_reshaped.permute(0, 2, 3, 1).reshape(batch.num_graphs, J * 3, F_target)
            npss_sum += calculate_npss(fk_pos_pred_npss, fk_pos_tgt_npss, reduction='sum').item()

            # Smoothness
            sm_1_sum += calculate_smoothness_loss(fk_pos_pred, fk_pos_tgt_reshaped, order=1, reduction='sum').item()
            sm_2_sum += calculate_smoothness_loss(fk_pos_pred, fk_pos_tgt_reshaped, order=2, reduction='sum').item()
            sm_3_sum += calculate_smoothness_loss(fk_pos_pred, fk_pos_tgt_reshaped, order=3, reduction='sum').item()

            # Foot contact
            height_eps = benchmark_foot_params["height_eps"]
            velocity_eps = benchmark_foot_params["velocity_eps"]

            foot_contact_tgt = compute_foot_contact(
                fk_pos=fk_pos_tgt_reshaped,
                foot_joint_indices=foot_joint_indices,
                contact_height_eps=height_eps,
                contact_velocity_eps=velocity_eps
            )

            foot_contact_pred = compute_foot_contact(
                fk_pos=fk_pos_pred,
                foot_joint_indices=foot_joint_indices,
                contact_height_eps=height_eps,
                contact_velocity_eps=velocity_eps
            )

            foot_contact_sum += calculate_foot_contact_loss(foot_contact_tgt, foot_contact_pred, reduction='sum').item()

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
            f"L2P value [{length_unit}]": l2p_value,
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
            f"Foot skating loss [{length_unit}/frame]": foot_skating_value
        }
    }


# Dataset
print("Loading dataset")
bechmark_dataset = GraphSkeletonDataset(
    root_dir=constants["benchmark_data_dir"],
    context_len_pre=config["context_len_pre"],
    context_len_post=config["context_len_post"],
    target_len=config["target_len"],
    step=constants["benchmark_dataset_step"]
)
print(f"Dataset ready with {len(bechmark_dataset)} samples")

benchmark_loader = DataLoader(bechmark_dataset, batch_size=constants["benchmark_dataset_batch_size"], shuffle=False)


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
os.makedirs(constants["benchmark_log_path"], exist_ok=True)
benchmark_log_path = constants["benchmark_log_path"] + filename + constants["log_suffix"]
if os.path.exists(benchmark_log_path):
    os.remove(benchmark_log_path)

log_str = lambda text: log_string(text=text, log_path=benchmark_log_path)


# Benchmarking
root_stats_path = constants["root_stats_path"] + filename + constants["root_stats_suffix"]
stats = np.load(root_stats_path)
root_mean = torch.tensor(stats["mean"], dtype=torch.float32).to(device).view(1, 1, 3)
root_std = torch.tensor(stats["std"], dtype=torch.float32).to(device).view(1, 1, 3)
print(f"Loaded root stats from: {root_stats_path}")

joint_names = bechmark_dataset.joint_names
parent_indices = bechmark_dataset.parent_indices.to(device)
offsets = bechmark_dataset.offsets.to(device)

print("Starting benchmark on test set")
results = run_benchmark(
    model=model,
    loader=benchmark_loader,
    offsets=bechmark_dataset.offsets.to(device),
    parent_indices=bechmark_dataset.parent_indices.to(device),
    root_mean=root_mean,
    root_std=root_std,
    n_samples=len(bechmark_dataset),
    benchmark_foot_params=constants["benchmark_foot_params"],
    joint_names=joint_names
)

log_str("\n--- Benchmark Results ---\n")
log_str(f"Model description: {config['description']}")

for metric_group, metric_values in results.items():
    log_str(f"\n{metric_group}:")
    for metric_name, value in metric_values.items():
        log_str(f"  {metric_name}: {' ' * (32 - len(metric_name))} {value:.7f}")

print()
