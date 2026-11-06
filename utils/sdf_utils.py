import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from utils.config import *


def query_feature_grid_3D(vertices, obj_sdf, sdf_centroid, sdf_scale):
    # 3D volume version.
    '''
    Bilinearly interpolates given positions in feature grid.
    - pos : (B, T, V, 3)
    - obj_sdf : (B x C x D x H x W)
    - sdf volume center: B x 3
    - sdf volume scale: B x 1

    Returns:
    - (B, C, T*V) feature interpolated to each position
    '''
    
    batch_size = vertices.shape[0]
    feat_dim = obj_sdf.shape[1]

    vertices = vertices.reshape(batch_size, -1, 3)
    num_vertices = vertices.shape[1]
    
    vertices = ((vertices - sdf_centroid[:, None].repeat((1, num_vertices, 1))) / sdf_scale[:, None].repeat((1, num_vertices, 1))) # scale to [-1,1]

    sdf_values = F.grid_sample(obj_sdf, vertices[:, :, [2, 1, 0]].view(-1, num_vertices, 1, 1, 3).float(), padding_mode='border', align_corners=True)                    

    return sdf_values.reshape(batch_size, feat_dim, num_vertices)


class OBJInfo:
    def __init__(self):
        
        self.OBJ_DIR = OBJ_DIR
        self._obj_sdf_cache = {}   # obj_id -> SDF
        self._obj_height_info_cache = {} # obj_id -> height_info

    def load_obj_sdf(self, obj_id):
        """Load SDF for obj_id, cache it."""
        if obj_id in self._obj_sdf_cache:
            return self._obj_sdf_cache[obj_id]

        sdf_path = os.path.join(self.OBJ_DIR, str(obj_id), "sdf_gradient.pkl")
        with open(sdf_path, 'rb') as f:
            object_sdf = pickle.load(f)

        sdf_grids = torch.from_numpy(object_sdf['grid']).squeeze().unsqueeze(0)  # 1xDxDxD
        gradient_grids = torch.from_numpy(object_sdf['gradient_grid']).permute(3, 0, 1, 2)  # 3xDxDxD
        sdf_centroid = torch.tensor(object_sdf['centroid']) # 3
        sdf_scale = torch.tensor(object_sdf['scale']).reshape(1) # 1
        
        self._obj_sdf_cache[obj_id] = (sdf_grids, gradient_grids, sdf_centroid, sdf_scale)
        
        return self._obj_sdf_cache[obj_id]


    def load_obj_height_info(self, obj_id):
        """Load height info for obj_id, cache it."""
        if obj_id in self._obj_height_info_cache:
            return self._obj_height_info_cache[obj_id]

        height_info_path = os.path.join(self.OBJ_DIR, str(obj_id), "height.pkl")
        with open(height_info_path, 'rb') as f:
            height_info = pickle.load(f)

        heightmap = torch.from_numpy(height_info['heightmap'])  # 64x64
        xz_min = torch.from_numpy(height_info['xz_min'])  # 2
        xz_max = torch.from_numpy(height_info['xz_max']) # 2   
        
        self._obj_height_info_cache[obj_id] = (heightmap, xz_min, xz_max)
        
        return self._obj_height_info_cache[obj_id]
    
    
    def load_batch_sdf(self, obj_ids, device):
        """
        Args:
            obj_ids: list of obj_id strings
            
        Returns:
            batch_sdf_grids: [B, 1, D, D, D]
            batch_gradient_grids: [B, 3, D, D, D]
            batch_sdf_centers: [B, 3]
            batch_sdf_scales: [B, 1]
        """
        batch_data = [self.load_obj_sdf(obj_id) for obj_id in obj_ids]
        sdf_grids, gradient_grids, sdf_centers, sdf_scales = zip(*batch_data)
        
        return (
            torch.stack(sdf_grids).to(device=device, dtype=torch.float32),
            torch.stack(gradient_grids).to(device=device, dtype=torch.float32),
            torch.stack(sdf_centers).to(device=device, dtype=torch.float32),
            torch.stack(sdf_scales).to(device=device, dtype=torch.float32)
        )
    
    def load_batch_height_info(self, obj_ids, device):
        """
        Args:
            obj_ids: list of obj_id strings
            
        Returns:
            batch_heightmaps: [B, H, W]
            batch_xz_mins: [B, 2]
            batch_xz_maxs: [B, 2]
        """
        batch_data = [self.load_obj_height_info(obj_id) for obj_id in obj_ids]
        heightmaps, xz_mins, xz_maxs = zip(*batch_data)
        
        return (
            torch.stack(heightmaps).to(device),
            torch.stack(xz_mins).to(device),
            torch.stack(xz_maxs).to(device)
        )
    
