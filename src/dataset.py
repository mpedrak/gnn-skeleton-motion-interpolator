import os
import torch

from torch_geometric.data import Data, Dataset

from .utils.bvh import parse_bvh_file, build_edge_index_from_parents, compute_root_deltas


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

        all_root_deltas = []

        for fname in self.files:
            filepath = os.path.join(root_dir, fname)
            cache_path = os.path.splitext(filepath)[0] + ".pt"
            if os.path.exists(cache_path):
                data = torch.load(cache_path)
                print(f"Loaded cached file: {cache_path}")
            else:
                root_pos, rot_6d, joint_names, parent_indices = parse_bvh_file(filepath)
                root_deltas = compute_root_deltas(root_pos)
                data = {
                    'root_deltas': root_deltas,
                    'rot_6d': rot_6d,
                    'joint_names': joint_names,
                    'parent_indices': parent_indices
                }
                torch.save(data, cache_path)
                print(f"Saved cache file: {cache_path}")

            self.cache[fname] = data
            all_root_deltas.append(data['root_deltas']) 

            frames = data['rot_6d'].shape[0]
            used_frames = context_len_pre + context_len_post + target_len
            for start in range(50, frames - used_frames, step):
                self.samples.append((fname, start))

        first_data = self.cache[self.files[0]]
        self.num_joints = len(first_data['joint_names'])
        # self.base_edge_index = build_edge_index_from_parents(first_data['parent_indices'])
        # F = context_len_pre + context_len_post
        # self.edge_index = build_spatio_temporal_edge_index(F, self.num_joints, self.base_edge_index)
        self.edge_index = build_edge_index_from_parents(first_data['parent_indices'])

        concat_root_deltas = torch.cat(all_root_deltas, dim=0) # [F, 3]
        self.root_mean = concat_root_deltas.mean(dim=0)            
        self.root_std = concat_root_deltas.std(dim=0)
        self.root_std = torch.where(self.root_std < 1e-8, torch.ones_like(self.root_std), self.root_std)
        

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        fname, start = self.samples[idx]
        data = self.cache[fname]
        post_ctx_start = start + self.context_len_pre + self.target_len

        first_part = data['rot_6d'][start : start + self.context_len_pre]
        second_part = data['rot_6d'][post_ctx_start : post_ctx_start + self.context_len_post]
        rot_6d_context = torch.cat([first_part, second_part], dim=0)  
        # x_feat = rot_6d_context.reshape(-1, 6) # [F, J, 6] -> [F * J, 6]
        x_feat = rot_6d_context.permute(1, 0, 2).reshape(self.num_joints, -1) # [F, J, 6] -> [J, F * 6]

        rot_6d_tgt = data['rot_6d'][start + self.context_len_pre : post_ctx_start]
        y_feat = rot_6d_tgt.permute(1, 0, 2).reshape(self.num_joints, -1) # [F, J, 6] -> [J, F * 6] 
   
        first_part = data['root_deltas'][start : start + self.context_len_pre] # [F, 3]
        second_part = data['root_deltas'][post_ctx_start : post_ctx_start + self.context_len_post]
        
        for i in [0, 1, 2]:
            first_part[0, i] = 0.0
            second_part[0, i] = 0.0

        root_ctx_raw = torch.cat([first_part, second_part], dim=0)

        root_tgt_raw = data['root_deltas'][start + self.context_len_pre : post_ctx_start]
       
        root_ctx_norm = (root_ctx_raw - self.root_mean) / self.root_std
        root_tgt_norm = (root_tgt_raw - self.root_mean) / self.root_std

        root_tgt_norm = root_tgt_norm.reshape(-1) # [F, 3] -> [F * 3]  
        root_ctx_norm = root_ctx_norm.reshape(-1)  

        return Data(
            x=x_feat,
            y=y_feat,
            edge_index=self.edge_index,
            root_ctx_norm=root_ctx_norm,
            root_tgt_norm=root_tgt_norm
        )
