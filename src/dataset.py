import os
import torch

from torch_geometric.data import Data, Dataset

from .utils.bvh import parse_bvh_file, build_edge_index_from_parents, compute_root_deltas, forward_kinematics_positions


class GraphSkeletonDataset(Dataset):
    def __init__(self, root_dir, context_len_pre, context_len_post, target_len, step):

        super().__init__()
        self.context_len_pre = context_len_pre
        self.context_len_post = context_len_post
        self.target_len = target_len

        self.files = [f for f in os.listdir(root_dir) if f.lower().endswith('.bvh')]
        if not self.files: raise FileNotFoundError(f"No BVH files found in: {root_dir}")
            
        self.cache = {}
        self.samples = []

        for fname in self.files:
            filepath = os.path.join(root_dir, fname)
            cache_path = os.path.splitext(filepath)[0] + ".pt"
            if os.path.exists(cache_path):
                data = torch.load(cache_path)
                print(f"Loaded cached file: {cache_path}")
            else:
                root_pos, rot_6d, joint_names, parent_indices, offsets = parse_bvh_file(filepath)
                fk_pos = forward_kinematics_positions(
                    offsets=offsets, 
                    parent_indices=parent_indices, 
                    root_pos=root_pos, 
                    rot_6d=rot_6d
                )
                root_pos_deltas = compute_root_deltas(root_pos)
                data = {
                    'root_pos_deltas': root_pos_deltas,
                    'rot_6d': rot_6d,
                    'joint_names': joint_names,
                    'parent_indices': parent_indices,
                    'offsets': offsets,
                    'fk_pos': fk_pos,
                    'root_pos_absolute': root_pos
                }
                torch.save(data, cache_path)
                print(f"Saved cache file: {cache_path}")

            self.cache[fname] = data

            frames = data['rot_6d'].shape[0]
            used_frames = context_len_pre + context_len_post + target_len
            for start in range(50, frames - used_frames, step):
                self.samples.append((fname, start))

        first_data = self.cache[self.files[0]]
        self.num_joints = len(first_data['joint_names'])
        self.joint_names = first_data['joint_names']
        self.edge_index = build_edge_index_from_parents(first_data['parent_indices'])
        self.parent_indices = first_data['parent_indices']
        self.offsets = first_data['offsets']
        

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        fname, start = self.samples[idx]
        data = self.cache[fname]
        tgt_start = start + self.context_len_pre
        post_ctx_start = tgt_start + self.target_len
        end = post_ctx_start + self.context_len_post
        
        # Context
        first_part = data['rot_6d'][start : tgt_start]
        second_part = data['rot_6d'][post_ctx_start : end]
        rot_6d_context = torch.cat([first_part, second_part], dim=0)  
        x_feat = rot_6d_context.permute(1, 0, 2).reshape(self.num_joints, -1) # [F, J, 6] -> [J, F * 6]

        first_part = data['root_pos_deltas'][start : tgt_start].clone() 
        second_part = data['root_pos_deltas'][post_ctx_start : end].clone()
        
        for i in [0, 1, 2]:
            first_part[0, i] = 0.0
            second_part[0, i] = 0.0

        root_ctx_raw = torch.cat([first_part, second_part], dim=0)

        # Target
        rot_6d_tgt = data['rot_6d'][tgt_start : post_ctx_start]
        y_feat = rot_6d_tgt.permute(1, 0, 2).reshape(self.num_joints, -1)
   
        root_tgt_absolute = data['root_pos_absolute'][tgt_start : post_ctx_start]
        
        fk_pos = data['fk_pos'][tgt_start : post_ctx_start]

        last_root_pos_absolute = data['root_pos_absolute'][tgt_start - 1].unsqueeze(0)

        return Data(
            x=x_feat,
            y=y_feat,
            edge_index=self.edge_index,
            root_pos_ctx=root_ctx_raw,
            root_pos_tgt=root_tgt_absolute,
            fk_pos=fk_pos,
            last_root_pos_absolute=last_root_pos_absolute
        )