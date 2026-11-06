import torch
import torch.nn.functional as F
from utils.bps_utils import canonicalize_poses_to_object_space
from utils.sdf_utils import query_feature_grid_3D
from pytorch3d.structures import Meshes
import imageio
import numpy as np
from utils.sample_frames import *

def approach_loss(x, y, target_d=0.7, tau=0.05):
        
    # Get the root locations of the last valid frame
    batch_idx = torch.arange(x.shape[0], device=x.device)
    last_frame_idx = (y['lengths'] - 1).long()  # [B,]
    last_root_locs = x[batch_idx, last_frame_idx, 0:1, :].unsqueeze(1)  # [B, 1, 1, 3]
    
    obj_coord_root_locs = canonicalize_poses_to_object_space(last_root_locs, y['obj_transform_trans'], y['obj_transform_rotation'], y['obj_scales']).squeeze(1) # [B, 1, 3]

    # compute distance from last frame root joint to surface points
    root_to_surf_dis = torch.cdist(obj_coord_root_locs, y['obj_sample_points']).squeeze()  # [B, S]

    # Softmin weights
    w = torch.softmax(-root_to_surf_dis / tau, dim=-1)
    soft_dis = (w * root_to_surf_dis).sum(dim=-1)
    loss = F.relu(soft_dis - target_d)  # penalize distance > target_d
    
    return loss


def ground_penetration_loss(x, y, pen_margin=0.0):

    ground_sdf = x[..., 1]  # ground is at y=0
    length_mask = y['mask'].squeeze() # [B, 1, 1, N] -> [B, N]
    
    J = x.shape[2]
    pene_loss = F.relu(-(ground_sdf + pen_margin)) * length_mask[:, :, None]
    valid_counts = length_mask.sum(dim=1) * J
    pene_loss = pene_loss.sum(dim=(1,2)) / valid_counts
    
    return pene_loss


def obj_penetration_loss(x, y, obj_info, pen_margin=0.0):
    
    length_mask = y['mask'].squeeze() # [B, 1, 1, N] -> [B, N]
    
    # get d_smal verts/joints in obj space
    obj_coord_joints = canonicalize_poses_to_object_space(x, y['obj_transform_trans'], y['obj_transform_rotation'], y['obj_scales']) # [B, N, J, 3] 
    
    # get sdf values & normals
    B, N, J, _ = obj_coord_joints.shape
    batch_sdf_grids, _, batch_sdf_centers, batch_sdf_scales = obj_info.load_batch_sdf(y['obj_id'], device=x.device)
    sdf = query_feature_grid_3D(obj_coord_joints, batch_sdf_grids, batch_sdf_centers, batch_sdf_scales).reshape(B, -1, N, J)  # [B, 1, N, J]
    sdf_vals = sdf[:, 0]
    
    # average pene loss over all joints
    pene_loss = F.relu(-(sdf_vals + pen_margin)) * length_mask[:, :, None]
    valid_counts = length_mask.sum(dim=1) * J
    pene_loss = pene_loss.sum(dim=(1,2)) / valid_counts

    return pene_loss


def adaptive_contact_loss(x, y, obj_info, contact_margin=0.0):
    
    foot_joint_idx = [10, 14, 20, 24]
    ground_dist = x[..., foot_joint_idx, 1]  # [B, N, J_foot]

    heightmaps, xz_mins, xz_maxs = obj_info.load_batch_height_info(y['obj_id'], device=x.device)
    
    # if foot in object xz range, compute y distance to object
    obj_coord_joints = canonicalize_poses_to_object_space(x, y['obj_transform_trans'], y['obj_transform_rotation'], y['obj_scales'])
    
    joints_foot = obj_coord_joints[..., foot_joint_idx, :]
    joints_xz = joints_foot[..., [0, 2]]
    in_range_x = (joints_xz[..., 0] >= xz_mins[:, None, None, 0]) & (joints_xz[..., 0] <= xz_maxs[:, None, None, 0])
    in_range_z = (joints_xz[..., 1] >= xz_mins[:, None, None, 1]) & (joints_xz[..., 1] <= xz_maxs[:, None, None, 1])
    in_obj_range = (in_range_x & in_range_z).detach()  # [B, N, J_foot]

    # normalize to [-1, 1], for grid_sample
    xz_range = (xz_maxs - xz_mins).clamp_min(1e-6)
    normalized_joints_xz = (joints_xz - xz_mins[:, None, None, :]) / xz_range[:, None, None, :]
    normalized_joints_xz = normalized_joints_xz * 2 - 1
    
    obj_heights = F.grid_sample(
        heightmaps.unsqueeze(1),  # [B, 1, H, W]
        normalized_joints_xz,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).squeeze(1)

    obj_dist = obj_coord_joints[..., foot_joint_idx, 1] - obj_heights
    
    length_mask = y['mask'].squeeze().bool()
    length_mask = length_mask.unsqueeze(-1).expand_as(in_obj_range) # [B, N, J_foot]

    def contact_loss(mask, dist):  
        # calculate k for each sample based on its valid frames
        valid_frames = (~mask).sum(dim=(1, 2))
        k = torch.clamp((valid_frames * 0.35).long(), min=4, max=600)  # [B,]
          
        dist[mask] = 1e6
        dist[dist <= 0] = 1e6  # ignore pene frames 

        # choose topk smallest distances (avoid wrongly process jump action)
        topk_min, _ = torch.topk(dist.reshape(x.shape[0], -1), k.max().item(), largest=False, dim=1)
        k_mask = torch.arange(k.max().item(), device=dist.device).unsqueeze(0) < k.unsqueeze(1)  # [B, k_max]
        topk_min[~k_mask] = 1e6
        
        valid_mask = topk_min < 1e6        
        loss = F.relu(topk_min - contact_margin) * valid_mask
        loss = loss.sum(dim=1) / valid_frames.clamp_min(1.0)
        
        return loss
    
    # ------ ground contact loss (for frames not in object range) ------
    ground_mask = ~(length_mask & (~in_obj_range))
    ground_loss = contact_loss(ground_mask, ground_dist)

    # ------ obj contact loss (for frames in object range) ------
    obj_mask = ~(length_mask & in_obj_range)
    obj_loss = contact_loss(obj_mask, obj_dist)
    
    return ground_loss + obj_loss


def temporal_smooth_loss(x, y):

    length_mask = y['mask'].squeeze()
    
    # acceleration smoothness
    vel = x[:, 1:, :, :] - x[:, :-1, :, :]
    acc = vel[:, 1:, :, :] - vel[:, :-1, :, :]
    loss_acc = (acc ** 2).sum(dim=(-1, -2)) * length_mask[:, 2:]
    loss_acc = loss_acc.sum(dim=1) / length_mask[:, 2:].sum(dim=1).clamp_min(1.0)
    
    return loss_acc


def video_SDS_loss(verts, faces, root_pos, y, 
                   mesh_bulider, render, video_SDS, 
                   lambda_vsd, weights, step, 
                   sample_frames=16, chunk_size=1):
        
    B = verts.shape[0]
    length_mask = y['mask'].squeeze()
    
    # sample k keyframes for each sequence
    keyframe_idx = uniform_sample(length_mask, k=sample_frames)
    batch_idx = torch.arange(B, device=verts.device)[:, None]
    verts_sampled = verts[batch_idx, keyframe_idx] # [B, k, V, 3]
    root_pos_sampled = root_pos[batch_idx, keyframe_idx].reshape(-1, 3) # [B*k, 3]

    # get smal+obj mesh in WCS
    meshes = mesh_bulider.build_scene_meshes(verts_sampled, faces, y) # B*k meshes
    # render video
    images = render.render_meshes_partition(meshes, root_pos_sampled) # [B*k, H, W, 3]

    # compute in chunks to save memory
    video_sds_loss_all = []
    
    for start in range(0, B, chunk_size): 

        end = min(start + chunk_size, B)

        # compute video SDS loss
        images_chunk = images[start*sample_frames:end*sample_frames] # [cs*k, H, W, 3]
        video_sds_loss_chunk = video_SDS(images_chunk.view(end-start, -1, 3, *images_chunk.shape[1:3]), y["text"][start:end], step)

        # chunk backward
        video_sds_loss_chunk = video_sds_loss_chunk * lambda_vsd * weights[start:end] / B
        video_sds_loss_chunk.mean().backward(retain_graph=True)
        
        # collect detach loss(only for logging)
        video_sds_loss_all.append(video_sds_loss_chunk.detach())

        del images_chunk, video_sds_loss_chunk
    
    return torch.cat(video_sds_loss_all, dim=0) * B / lambda_vsd

