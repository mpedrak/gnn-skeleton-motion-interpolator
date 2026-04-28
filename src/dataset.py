import os
import torch

from torch_geometric.data import Data, Dataset

from .utils.bvh import parse_bvh_file, build_edge_index, forward_kinematics_pos, print_skeleton_hierarchy


class GraphSkeletonDataset(Dataset):
    def __init__(self, data_params, context_len_pre, context_len_post, target_len):

        super().__init__()
        self.context_len_pre = context_len_pre
        self.context_len_post = context_len_post
        self.target_len = target_len

        self.cache = {}
        self.samples = []

        self.dir_info = []
       
        print("Loading dataset")
        
        for data in data_params:

            root_dir = data["dir"]
            step = data["step"]
            skip_start = data["skip_start"]

            print(f"Processing directory: {root_dir}")

            files = [f for f in os.listdir(root_dir) if f.lower().endswith('.bvh')]
            if not files: raise FileNotFoundError(f"No BVH files found in: {root_dir}")

            prev_sample_count = len(self.samples)

            for fname in files: 

                filepath = os.path.join(root_dir, fname)
                cache_path = os.path.splitext(filepath)[0] + ".pt"
                if os.path.exists(cache_path):
                    data = torch.load(cache_path)
                    print(f"Loaded cached file: {cache_path}")
                else:
                    root_pos, rot_6d, joint_names, parent_indices, offsets, _ = parse_bvh_file(filepath)
                    fk_pos, global_3x3_rots = forward_kinematics_pos(
                        offsets=offsets, 
                        parent_indices=parent_indices, 
                        root_pos=root_pos, 
                        rot_6d=rot_6d
                    )
                    edge_index = build_edge_index(parent_indices)
                    data = {
                        'rot_6d': rot_6d,
                        'fk_pos': fk_pos,
                        'root_pos': root_pos,
                        'joint_names': joint_names,
                        'parent_indices': parent_indices,
                        'offsets': offsets,
                        'edge_index': edge_index,
                        'global_3x3_rots': global_3x3_rots
                    }
                    torch.save(data, cache_path)
                    print(f"Saved and loaded cache file: {cache_path}")

                self.cache[fname] = data

                frames = data['rot_6d'].shape[0]
                used_frames = context_len_pre + context_len_post + target_len
                for start in range(skip_start, frames - used_frames, step):
                    self.samples.append((fname, start))

            first_data = self.cache[files[0]]
            print(f"Finished processing: {root_dir}")

            self.dir_info.append({
                "root_dir": root_dir,
                "num_joints": len(first_data['joint_names']),
                "num_samples": len(self.samples) - prev_sample_count,
                "joint_names": first_data['joint_names'],
                "parent_indices": first_data['parent_indices']
            })

        print(f"\nDataset ready with {len(self.samples)} samples")
        
        for dir in self.dir_info:
            print(f"Directory: {dir['root_dir']}")
            print(f"Number of joints: {dir['num_joints']}, samples from this dir: {dir['num_samples']}")
            print_skeleton_hierarchy(joint_names=dir["joint_names"], parent_indices=dir["parent_indices"])
            print()


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):

        fname, start = self.samples[idx]
        data = self.cache[fname]
        tgt_start = start + self.context_len_pre
        post_ctx_start = tgt_start + self.target_len
        end = post_ctx_start + self.context_len_post
        num_joints = len(data['joint_names'])
        
        # Context
        first_part = data['rot_6d'][start : tgt_start]
        second_part = data['rot_6d'][post_ctx_start : end]
        rot_6d_ctx = torch.cat([first_part, second_part], dim=0) # [F, J, 6]  
       
        first_part = data['root_pos'][start : tgt_start]
        second_part = data['root_pos'][post_ctx_start : end]
        first_ctx_root_pos = first_part[0].clone()
        first_part = first_part.clone() - first_ctx_root_pos
        second_part = second_part.clone() - first_ctx_root_pos
        root_pos_ctx = torch.cat([first_part, second_part], dim=0) # [F, 3]

        # Target
        rot_6d_tgt = data['rot_6d'][tgt_start : post_ctx_start] # [F, J, 6]
        root_pos_tgt = data['root_pos'][tgt_start : post_ctx_start] # [F, 3]
        fk_pos_tgt = data['fk_pos'][tgt_start : post_ctx_start] # [F, J, 3]
 
        root_pos_for_lerp = torch.stack([data['root_pos'][tgt_start - 1], data['root_pos'][post_ctx_start]], dim=0) # [2, 3]
        rot_6d_for_slerp = torch.stack([data['rot_6d'][tgt_start - 1], data['rot_6d'][post_ctx_start]], dim=0) # [2, J, 6]

        global_3x3_rots_tgt = data['global_3x3_rots'][tgt_start : post_ctx_start] # [F, J, 3, 3]
        
        # Permute node level features for PyG batch
        x_feat = rot_6d_ctx.permute(1, 0, 2).reshape(num_joints, -1) # -> [J, F * 6]
        y_feat = rot_6d_tgt.permute(1, 0, 2).reshape(num_joints, -1) # -> [J, F * 6]
        fk_pos_tgt = fk_pos_tgt.permute(1, 0, 2) # -> [J, F, 3]
        rot_6d_for_slerp = rot_6d_for_slerp.permute(1, 0, 2) # -> [J, 2, 6]
        global_3x3_rots_tgt = global_3x3_rots_tgt.permute(1, 0, 2, 3) # -> [J, F, 3, 3]

        return Data(
            x=x_feat,
            y=y_feat,
            root_pos_ctx=root_pos_ctx,
            root_pos_tgt=root_pos_tgt,
            fk_pos_tgt=fk_pos_tgt,
            root_pos_for_lerp=root_pos_for_lerp,
            rot_6d_for_slerp=rot_6d_for_slerp,
            global_3x3_rots_tgt=global_3x3_rots_tgt,
            offsets=data['offsets'],
            edge_index=data['edge_index'],
            parent_indices=data['parent_indices'],
        )
