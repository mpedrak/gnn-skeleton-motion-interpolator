from interface.general_interface import GeneralInterface
# https://github.com/kottajl/blender-plug-in-for-modifying-3D-animations/blob/main/interface/general_interface.py


import torch
import json
import numpy as np
import sys

from torch_geometric.data import Data
from scipy.spatial.transform import Rotation as R
from pathlib import Path


src_path = Path(__file__).parent.resolve() / "src"
sys.path.append(str(src_path))

from model import SkeletalMotionInterpolator
from utils.rotation import rot_6d_to_euler, rot_3x3_to_rot_6d
from utils.bvh import build_edge_index, compute_lerp, forward_kinematics_pos, global_to_local_rot
from utils.rotation import rot_6d_to_rot_3x3, rot_3x3_to_rot_6d, compute_slerp
from utils.various import load_configs, set_global_seed

sys.path.remove(str(src_path))
print("Successfully loaded model source files")


main_dir = Path(__file__).parent.resolve() 
main_dir = str(main_dir) + "/"
print(f"Main directory: {main_dir}")
config_dir = main_dir + "configs/"


@torch.no_grad()
def predict_gap(model, device, rot_6d, root_pos, parent_indices, context_len_pre, context_len_post, target_len, gap_start, offsets, inner_rots, root_pos_delta_mode, rotations_delta_mode):
    
    J = rot_6d.shape[1]
    second_start = gap_start + target_len
    end = second_start + context_len_post
    first_start = gap_start - context_len_pre

    # Rotations and offsets
    first_part_rot = rot_6d[first_start : gap_start]
    second_part_rot = rot_6d[second_start : end]
    rot_ctx = np.concatenate([first_part_rot, second_part_rot], axis=0) # [F, J, 6]

    if inner_rots == "global":
        first_part_root_pos = root_pos[first_start : gap_start]
        second_part_root_pos = root_pos[second_start : end] 
        root_pos_ctx = torch.cat([first_part_root_pos, second_part_root_pos], dim=0).to(device) 
        rot_ctx_tensor = torch.from_numpy(rot_ctx).to(device, dtype=torch.float32)
        offsets_tensor = offsets.clone().detach().to(device, dtype=torch.float32)
        parents_tensor = parent_indices.clone().detach().to(device)
        _, global_3x3_rots = forward_kinematics_pos(
            offsets=offsets_tensor,
            parent_indices=parents_tensor,
            root_pos=root_pos_ctx,
            rot_6d=rot_ctx_tensor,
            local_rots=True
        )
        rot_ctx = rot_3x3_to_rot_6d(global_3x3_rots)
        rot_ctx = rot_ctx.cpu().numpy()
        
    x_feat = torch.tensor(rot_ctx, dtype=torch.float32).permute(1, 0, 2).reshape(J, -1) # [J, F * 6]
    offsets_tensor = torch.tensor(offsets, dtype=torch.float32) if not isinstance(offsets, torch.Tensor) else offsets.clone()
    parent_tensor = torch.tensor(parent_indices) if not isinstance(parent_indices, torch.Tensor) else parent_indices
    offsets_tensor[parent_tensor == -1] = 0.0
    bone_lengths = torch.linalg.norm(offsets_tensor, dim=1, keepdim=True)
    x_feat = torch.cat([x_feat, bone_lengths], dim=1) # [J, F * 6 + 1]

    # Root positions
    first_part_root_pos = root_pos[first_start : gap_start]
    second_part_root_pos = root_pos[second_start : end] 
    first_ctx_root_pos = first_part_root_pos[0].clone()
    first_part_root_pos = first_part_root_pos - first_ctx_root_pos
    second_part_root_pos = second_part_root_pos - first_ctx_root_pos 
    root_pos_ctx = torch.cat([first_part_root_pos, second_part_root_pos], dim=0).to(device) 

    # Graph
    edge_index = build_edge_index(parent_indices)

    data = Data(
        x=x_feat,
        edge_index=edge_index,
        root_pos_ctx=root_pos_ctx
    )
    data = data.to(device)
    out = model(data)
    
    # Reshape rotations
    rot_pred_delta = out["rot"]
    rot_pred_delta = rot_pred_delta.view(J, target_len, 6).permute(1, 0, 2).contiguous() # [J, F, 6] -> [F, J, 6]  

    # Reconstruct rotations from deltas
    slerp_start_6d = rot_6d[gap_start - 1].clone().detach().to(device, dtype=torch.float32)

    if rotations_delta_mode == "linear":
        slerp_end_6d = rot_6d[second_start].clone().detach().to(device, dtype=torch.float32)
        if inner_rots == "global":
            slerps_rots = torch.cat([slerp_start_6d.unsqueeze(0), slerp_end_6d.unsqueeze(0)], dim=0) # [2, J, 6]
            slerps_root_pos = torch.cat([root_pos[gap_start - 1].unsqueeze(0), root_pos[second_start].unsqueeze(0)], dim=0).to(device) # [2, 3]
            offsets_tensor = offsets.clone().detach().to(device, dtype=torch.float32)
            parents_tensor = parent_indices.clone().detach().to(device)
            _, global_3x3_rots = forward_kinematics_pos(
                offsets=offsets_tensor,
                parent_indices=parents_tensor,
                root_pos=slerps_root_pos,
                rot_6d=slerps_rots,
                local_rots=True
            )
            global_6d_rots = rot_3x3_to_rot_6d(global_3x3_rots)
            slerp_start_6d = global_6d_rots[0]
            slerp_end_6d = global_6d_rots[1]

        rot_slerp = compute_slerp(slerp_start_6d, slerp_end_6d, target_len)

    elif rotations_delta_mode == "last_frame":
        rot_slerp = slerp_start_6d.unsqueeze(0).expand(target_len, -1, -1) # [F, J, 6]
        rot_slerp = rot_6d_to_rot_3x3(rot_slerp)

    if rotations_delta_mode == "none":
        rot_pred = rot_6d_to_rot_3x3(rot_pred_delta)
    else:
        rot_pred_delta = rot_6d_to_rot_3x3(rot_pred_delta) 
        rot_pred = torch.matmul(rot_pred_delta, rot_slerp)

    if inner_rots == "global":
        parents_tensor = parent_indices.clone().detach().to(device)
        rot_pred = global_to_local_rot(
                global_rots=rot_pred,
                parent_indices=parents_tensor
            )  
        
    rot_pred = rot_3x3_to_rot_6d(rot_pred) 

    # Reconstruct root positions from deltas
    root_pos_delta_pred = out["root_pos"]
    root_pos_delta_pred = root_pos_delta_pred.view(1, -1).view(target_len, 3) # [F, 3]
    lerp_start_pos = root_pos[gap_start - 1]

    if root_pos_delta_mode == "linear":
        lerp_end_pos = root_pos[second_start]
        root_pos_lerp = compute_lerp(lerp_start_pos, lerp_end_pos, target_len)
    elif root_pos_delta_mode == "last_frame":
        root_pos_lerp = lerp_start_pos.unsqueeze(0).expand(target_len, -1).to(device) # [F, 3]

    if root_pos_delta_mode == "none":
        root_pos_pred = root_pos_delta_pred
    else:
        root_pos_pred = root_pos_delta_pred + root_pos_lerp.to(device)        

    return rot_pred.cpu(), root_pos_pred.cpu()


class ModelInterface(GeneralInterface):

    '''
    General Interface Implementation for the AI Animation Bridge plugin 
    https://github.com/kottajl/blender-plug-in-for-modifying-3D-animations/tree/main
    '''

    def get_additional_infer_params(self) -> list[tuple[type, str, str]]:
        return [
                (torch.device, "Device", "Select device to compute on"),
                (str, "Version", "Version of the model to use (eg. 45_b for multiple skeletons or 51_c for lafan skeleton)"),         
            ]


    def check_frame_range(self, start_frame, end_frame, scene_start_frame, scene_end_frame, **kwargs) -> tuple[bool, str]:
        version = kwargs.get("Version", None) 
        if version is None: return (False, "Version of the model must be specified.")

        filename = "v_" + version
        config = load_configs([filename], config_dir=config_dir)[0]

        t_pre = config["context_len_pre"] 
        t_gen = config["target_len"]
        t_post = config["context_len_post"]

        if start_frame < scene_start_frame + t_pre:  return (False, f"Must be at least {t_pre} frames before selected range.") 
        if end_frame + t_post > scene_end_frame:  return (False, f"Must be at least {t_post} frames after selected range.") 
        if end_frame - start_frame + 1 != t_gen : return (False, f"Must be exactly {t_gen} frames in selected range.") 

        return (True, "")


    def is_skeleton_supported(self, skeleton, **kwargs) -> bool:
        version = kwargs.get("Version", None) 
        if version is None: return (False, "Version of the model must be specified.")
        
        filename = "v_" + version
        constants = load_configs(["constants"], config_dir=config_dir)[0]
        
        skeletons_path = main_dir + constants["skeletons_path"] + filename + constants["skeletons_suffix"]
        with open(skeletons_path, 'r') as f:
            loaded_list = json.load(f)

        model_supported_skeletons = {tuple(tuple(joint) for joint in skeleton) for skeleton in loaded_list}

        sorted_skeleton = tuple(tuple(joint) for joint in sorted(skeleton))

        if sorted_skeleton not in model_supported_skeletons: print(f"WARNING: this skeleton topology was not used during training")

        return True
    

    def infer_anim(self, anim_data, start_frame, end_frame, **kwargs):
        # Model arguments
        device = kwargs.get("Device", "cpu") 
        version = kwargs.get("Version", None) 
        if version is None: return (False, "Version of the model must be specified.")
        
        filename = "v_" + version
        config, constants = load_configs([filename, "constants"], config_dir=config_dir)
        print(f"Loaded config for version: {filename}")

        set_global_seed(config["seed"])
        print(f"Global seed set to: {config['seed']}")

        # Model
        model = SkeletalMotionInterpolator(
            context_len_pre=config["context_len_pre"],
            context_len_post=config["context_len_post"],
            target_len=config["target_len"],
            rot_gnn_params=config["rot_gnn_params"],
            root_pos_mlp_params=config["root_pos_mlp_params"]
        )
        model = model.to(device)

        model_path = main_dir + constants["models_path"] + filename + constants["models_suffix"]
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {model_path}")

        # Data preparation
        context_len_pre = config["context_len_pre"]
        context_len_post = config["context_len_post"]
        target_len = config["target_len"]

        rotations = anim_data["rotations"]
        rot_6d = rot_3x3_to_rot_6d(rotations)

        positions = anim_data["positions"]
        root_pos = positions[:, 0, :]
        offsets = anim_data["offsets"]
        parent_indices = anim_data["parents"]

        rot_6d = torch.tensor(rot_6d, dtype=torch.float32)
        root_pos = torch.tensor(root_pos, dtype=torch.float32)
        offsets = torch.tensor(offsets, dtype=torch.float32)
        parent_indices = torch.tensor(parent_indices, dtype=torch.int64)
        
        gap_start_frame = start_frame

        print("Data preparation complete")

        # Prediction
        model.eval()
        
        with torch.no_grad():
            rot_pred, root_pred = predict_gap(
                model=model,
                device=device,
                rot_6d=rot_6d,
                root_pos=root_pos,
                parent_indices=parent_indices,
                context_len_pre=context_len_pre,
                context_len_post=context_len_post,
                target_len=target_len,
                gap_start=gap_start_frame,
                offsets=offsets,
                inner_rots=config["inner_rots"],
                root_pos_delta_mode=config["root_pos_delta_mode"],
                rotations_delta_mode=config["rotations_delta_mode"]
            )

        print("Prediction complete")

        matrxices_pred = rot_6d_to_rot_3x3(rot_pred) # [target_len, J, 3, 3]
        matrxices_pred_np = matrxices_pred.detach().cpu().numpy()
        q = R.from_matrix(matrxices_pred_np).as_quat() 
        q_pred = q[..., [3, 0, 1, 2]]
        r_pred_np = R.from_quat(q_pred).as_matrix()
        r_pred = torch.from_numpy(r_pred_np).to(device=rot_pred.device, dtype=rot_pred.dtype)
        rot_6d = rot_3x3_to_rot_6d(r_pred)
        euler_deg = rot_6d_to_euler(rot_6d, order="XYZ", degrees=True)
        euler_deg = euler_deg[..., [2, 1, 0]] # [target_len, J, 3]

        root_pos_pred = root_pred.unsqueeze(1) # [target_len, 1, 3]
        positions_pred = torch.cat([root_pos_pred, offsets.unsqueeze(0).expand(target_len, -1, -1)], dim=1) # [target_len, J, 3]
        positions_pred = positions_pred.cpu().numpy()

        print("All done")

        return positions_pred, euler_deg
