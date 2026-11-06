import numpy as np
import torch
from data_loaders.common.quaternion import *

def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def get_verts_from_data(x, scale_betas, smal_model, inv_transform_th=None):
    
    if x.dim() == 4: # train data
        x = x[:, :, 0, :]
        x = inv_transform_th(x.permute((0, 2, 1))) # [batch_size, max_frames, njoints] in real scale
    N = x.shape[1]

    _, root_pos = recover_root_rot_pos(x)
    motions_rot, root_pos = x[:, :, 109:319].reshape(-1, 35, 6), root_pos.reshape(-1, 3)  # (B*N, J, 6) (B*N, 3)
    
    scale, betas, betas_limbs = scale_betas[:, 0:1], scale_betas[:, 1:31], scale_betas[:, 31:]
    
    verts, joints = smal_model(beta=betas.repeat_interleave(N, dim=0), 
                               betas_limbs=betas_limbs.repeat_interleave(N, dim=0), 
                               scale=scale.repeat_interleave(N, dim=0), 
                               trans=root_pos, 
                               pose=cont6d_to_matrix(motions_rot),
                               )
    
    verts = verts.reshape(x.shape[0], -1, *verts.shape[-2:]) # (B, N, V, 3)
    joints = joints.reshape(x.shape[0], -1, *joints.shape[-2:]) # (B, N, J, 3)
    root_pos = root_pos.reshape(x.shape[0], -1, 3) # (B, N, 3)
    
    return verts, joints, root_pos

