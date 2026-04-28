import torch
import os
import argparse
import numpy as np

from torch_geometric.loader import DataLoader 
from tqdm import tqdm

from src.dataset import GraphSkeletonDataset
from src.model import SkeletalMotionInterpolator
from src.utils.metrics import geodesic_rotation_loss, calculate_l2p, calculate_l2q, calculate_npss
from src.utils.metrics import calculate_smoothness_loss
from src.utils.bvh import forward_kinematics_pos_dense_batch, compute_lerp_batch
from src.utils.various import load_configs, log_string, set_global_seed
from src.utils.rotation import rot_6d_to_rot_3x3, compute_slerp_batch, rot_3x3_to_rot_6d, rot_6d_to_quat_torch


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
args = parser.parse_args()
filename = args.config

config, constants = load_configs([filename, "constants"])
print(f"Loaded config: {filename}")
print(f"Model description: {config['description']}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

set_global_seed(constants["seed"])

if device == "cuda":
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True


# Evaluate function
def run_benchmark(model, loader, n_samples):
           
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

    total_joints = 0

    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Test", leave=False):
            
            batch = batch.to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16): 
                out = model(batch)

            out["rot"] = out["rot"].float()
            out["root_pos"] = out["root_pos"].float()

            rot_pred_delta = out["rot"] 
            N_total, Fx6 = rot_pred_delta.shape
            F_target = Fx6 // 6 
            rot_pred_delta = rot_pred_delta.view(N_total, F_target, 6)
            total_joints += N_total

            # Rotations reconstructing from deltas
            rot_6d_for_slerp = batch.rot_6d_for_slerp # [N_total, 2, 6]
            slerp_start_6d = rot_6d_for_slerp[:, 0, :]
            slerp_end_6d = rot_6d_for_slerp[:, 1, :]
            rot_slerp = compute_slerp_batch(slerp_start_6d, slerp_end_6d, F_target) 
            
            rot_pred_delta_3x3 = rot_6d_to_rot_3x3(rot_pred_delta)
            rot_pred_3x3 = torch.matmul(rot_pred_delta_3x3, rot_slerp) # [N_total, F_target, 3, 3]
            
            # Position reconstruction from deltas and forward kinematics
            root_pos_delta_pred = out['root_pos']
            root_pos_delta_pred = root_pos_delta_pred.view(batch.num_graphs, F_target, 3)
            root_pos_for_lerp = batch.root_pos_for_lerp.view(batch.num_graphs, 2, 3) # [B, 2, 3]
            lerp_start_pos = root_pos_for_lerp[:, 0, :] 
            lerp_end_pos = root_pos_for_lerp[:, 1, :]
            root_pos_lerp = compute_lerp_batch(lerp_start_pos, lerp_end_pos, F_target)
            root_pos_pred = root_pos_delta_pred + root_pos_lerp

            fk_pos_pred, global_3x3_rots_pred = forward_kinematics_pos_dense_batch( 
                offsets=batch.offsets,
                parent_indices=batch.parent_indices,
                root_pos=root_pos_pred,
                rot_3x3=rot_pred_3x3,
                batch_index=batch.batch
            ) # [N_total, F_target, 3], [N_total, F_target, 3, 3]

            # Rotation metrics
            global_rot_tgt_3x3 = batch.global_3x3_rots_tgt.view(N_total, F_target, 3, 3)
            rot_geo_1_sum += rot_geo_1(global_3x3_rots_pred, global_rot_tgt_3x3).item()
            rot_geo_2_sum += rot_geo_2(global_3x3_rots_pred, global_rot_tgt_3x3).item()

            # L2Q
            global_rot_pred_6d = rot_3x3_to_rot_6d(global_3x3_rots_pred)
            global_rot_tgt_6d = rot_3x3_to_rot_6d(global_rot_tgt_3x3)
            l2q_sum += calculate_l2q(global_rot_pred_6d, global_rot_tgt_6d, reduction='sum').item()

            # Root positions metrics
            root_pos_tgt = batch.root_pos_tgt.view(batch.num_graphs, F_target, 3)
            root_pos_l1_sum += l1(root_pos_pred, root_pos_tgt).item()
            root_pos_l2_sum += l2(root_pos_pred, root_pos_tgt).item()

            # All positions metrics
            fk_pos_tgt = batch.fk_pos_tgt
            fk_pos_tgt_flat = fk_pos_tgt.view(N_total, -1)
            fk_pos_pred_flat = fk_pos_pred.view(N_total, -1)
            all_pos_l1_sum += l1(fk_pos_pred_flat, fk_pos_tgt_flat).item()
            all_pos_l2_sum += l2(fk_pos_pred_flat, fk_pos_tgt_flat).item()

            # L2P
            l2p_sum += calculate_l2p(fk_pos_pred, fk_pos_tgt, reduction='sum').item()

            # NPSS
            global_quat_pred = rot_6d_to_quat_torch(global_rot_pred_6d)
            global_quat_tgt = rot_6d_to_quat_torch(global_rot_tgt_6d)

            for t in range(1, F_target):
                dot_pred = (global_quat_pred[:, t, :] * global_quat_pred[:, t-1, :]).sum(dim=-1, keepdim=True)
                global_quat_pred[:, t, :] = torch.where(dot_pred < 0, -global_quat_pred[:, t, :], global_quat_pred[:, t, :])
                dot_tgt = (global_quat_tgt[:, t, :] * global_quat_tgt[:, t-1, :]).sum(dim=-1, keepdim=True)
                global_quat_tgt[:, t, :] = torch.where(dot_tgt < 0, -global_quat_tgt[:, t, :], global_quat_tgt[:, t, :])

            quat_pred_npss = global_quat_pred.permute(0, 2, 1)
            quat_tgt_npss = global_quat_tgt.permute(0, 2, 1)
            npss_sum += calculate_npss(quat_pred_npss, quat_tgt_npss, reduction='sum').item()

            # Smoothness metrics
            sm_1_sum += calculate_smoothness_loss(fk_pos_pred, fk_pos_tgt, order=1, reduction='sum').item()
            sm_2_sum += calculate_smoothness_loss(fk_pos_pred, fk_pos_tgt, order=2, reduction='sum').item()
            sm_3_sum += calculate_smoothness_loss(fk_pos_pred, fk_pos_tgt, order=3, reduction='sum').item()


    geo_rot_1 = rot_geo_1_sum / (total_joints * F_target)
    geo_rot_1_deg = geo_rot_1 * (180.0 / np.pi)

    geo_rot_2 = np.sqrt(rot_geo_2_sum / (total_joints * F_target))
    geo_rot_2_deg = geo_rot_2 * (180.0 / np.pi)

    root_pos_mae = root_pos_l1_sum / (n_samples * F_target * 3)
    
    root_pos_rmse = np.sqrt(root_pos_l2_sum / (n_samples * F_target * 3))

    all_pos_mae = all_pos_l1_sum / (total_joints * F_target  * 3)
   
    all_pos_rmse = np.sqrt(all_pos_l2_sum / (total_joints * F_target  * 3))

    l2p_value = l2p_sum / (total_joints * F_target)

    l2q_value = l2q_sum / (total_joints * F_target)

    npss_value = npss_sum / (total_joints * 4)

    sm_1_value = sm_1_sum / (total_joints * (F_target - 1) * 3)
    sm_2_value = sm_2_sum / (total_joints * (F_target - 2) * 3)
    sm_3_value = sm_3_sum / (total_joints * (F_target - 3) * 3)

    length_unit = constants["length_unit"]

    return {
        "Per joint" : {
            f"L2P value [{length_unit}]": l2p_value,
            "L2Q value": l2q_value,
            "Rotation MAE [deg]": geo_rot_1_deg,
            "Rotation RMSE [deg]": geo_rot_2_deg,
        },
        "Per joint quaternion channel" : {
            "NPSS value": npss_value,
        },
        "Per joint axis channel": {
            f"Root position MAE [{length_unit}]": root_pos_mae,
            f"Root position RMSE [{length_unit}]": root_pos_rmse,
            f"All positions MAE [{length_unit}]": all_pos_mae,
            f"All positions RMSE [{length_unit}]": all_pos_rmse,
            f"Velocity loss [{length_unit}/frame]": sm_1_value,
            f"Acceleration loss [{length_unit}/frame^2]": sm_2_value,
            f"Jerk loss [{length_unit}/frame^3]": sm_3_value    
        }
    }


# Dataset
bechmark_dataset = GraphSkeletonDataset(
    data_params=config["test_data_params"],
    context_len_pre=config["context_len_pre"],
    context_len_post=config["context_len_post"],
    target_len=config["target_len"]
)

benchmark_loader = DataLoader(bechmark_dataset, batch_size=config["test_batch_size"], shuffle=False)


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


print("Starting benchmark on test set")
results = run_benchmark(
    model=model,
    loader=benchmark_loader,
    n_samples=len(bechmark_dataset)
)

log_str("\n--- Benchmark Results ---\n")
log_str(f"Model description: {config['description']}")

for metric_group, metric_values in results.items():
    log_str(f"\n{metric_group}:")
    for metric_name, value in metric_values.items():
        log_str(f"  {metric_name}: {' ' * (32 - len(metric_name))} {value:10.5f}")

print()
