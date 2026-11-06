import numpy as np
import torch
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from utils.config import *

ROOT_CODE_DIR = os.path.dirname(os.path.dirname(__file__))
_BPS_SET_CACHE = {}
_BPS_SET_DEVICE_CACHE = {}

# Sample uniformly in the unit ball (random) | 
def bps_gen_ball_inside(n_bps=SAMPLE_POINTS_NUM, random_seed=100): # with 1 radius around the center of the object.
    np.random.seed(random_seed)
    x = np.random.normal(size=[n_bps, 3])
    x_norms = np.sqrt(np.sum(np.square(x), axis=1)).reshape([-1, 1])
    x_unit = x / x_norms  # points on the unit ball surface
    r = np.random.uniform(size=[n_bps, 1])
    u = np.power(r, 1.0 / 3)
    basis_set = 1 * x_unit * u  # basic set coordinates, [n_bps, 3]
    return basis_set


# generate bps template
def load_bps_set(n_bps, save_dir=OBJ_DIR, device=None):
    os.makedirs(save_dir, exist_ok=True)
    bps_set_path = os.path.join(save_dir, 'bps_set.npy')
    cache_key = (os.path.abspath(save_dir), int(n_bps))

    if cache_key not in _BPS_SET_CACHE:
        if not os.path.exists(bps_set_path):
            print(f'Generating and saving bps set to [{bps_set_path}]...')
            bps_set = bps_gen_ball_inside(n_bps, random_seed=100)
            with open(bps_set_path, 'wb') as f:
                np.save(f, bps_set)
        else:
            with open(bps_set_path, 'rb') as f:
                bps_set = np.load(f)
        _BPS_SET_CACHE[cache_key] = torch.from_numpy(bps_set).float()

    bps_set = _BPS_SET_CACHE[cache_key]
    if device is None:
        return bps_set

    device_key = cache_key + (str(device),)
    cached = _BPS_SET_DEVICE_CACHE.get(device_key)
    if cached is None or cached.device != device:
        cached = bps_set.to(device)
        _BPS_SET_DEVICE_CACHE[device_key] = cached
    return cached # [n_bps, 3]


def get_one_directional_bps_feature(bps_set, point_set, return_distance_vector=False):

    assert len(bps_set.shape) == 2 and len(point_set.shape) == 4
    B, N, J, _ = point_set.shape
    point_set = point_set.reshape(-1, J, 3)
    bps_set = bps_set.float().to(point_set.device)    
    bps_set = bps_set.unsqueeze(0).expand(B*N, -1, -1)
    
    distances_A_to_B = torch.cdist(bps_set, point_set) 
    nearest_distances_A_to_B, nearest_indices_A_to_B = torch.min(distances_A_to_B, dim=-1)

    if return_distance_vector:
        batch_indices = torch.arange(
            nearest_indices_A_to_B.shape[0],
            device=point_set.device,
        ).unsqueeze(-1).expand_as(nearest_indices_A_to_B)
        nearest_points = point_set[batch_indices, nearest_indices_A_to_B]
        return_value = nearest_points - bps_set
    else:
        return_value = nearest_distances_A_to_B
    
    return return_value.view(B, N, -1), nearest_indices_A_to_B.view(B, N, -1)


def canonicalize_poses_to_object_space(input_joints, obj_transl, obj_rot_mat, obj_scale):
    
    B, N, J, _ = input_joints.shape
    pts = input_joints.reshape(B, -1, 3)
    pts_centered = pts - obj_transl[:, None, :]
    
    # rotate by inverse of object rotation
    rot_inv = torch.inverse(obj_rot_mat)
    pts_obj = torch.bmm(rot_inv, pts_centered.permute(0,2,1)).permute(0,2,1)
    pts_obj /= obj_scale[:, None, None]
    
    return pts_obj.reshape(B, N, J, 3)


def get_bps_pose_feature(world_poses, y):
    
    # world_poses: [B, N, J, 3], joints location in WCS
    
    bps_set = load_bps_set(n_bps=SAMPLE_POINTS_NUM, device=world_poses.device)
        
    obj_tranform_trans = y['obj_transform_trans'].to(world_poses.device) # [B, 3]
    obj_tranform_rotation = y['obj_transform_rotation'].to(world_poses.device) # [B, 3, 3]
    obj_scales = y['obj_scales'].to(world_poses.device) # [B, ]
    obj_space_poses = canonicalize_poses_to_object_space(world_poses, obj_tranform_trans, obj_tranform_rotation, obj_scales) # [B, N, J, 3], joints location in obj-local space 

    bps_to_joint_dis, bps_to_joint_idx = get_one_directional_bps_feature(bps_set, obj_space_poses) # [B, N, n_bps]

    return bps_to_joint_dis, bps_to_joint_idx
