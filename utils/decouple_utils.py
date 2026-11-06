import torch
import math

def _axis_angle_rotation(axis: str, angle):
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def apply_rt_to_motion(motion, raw_out):
    """
    motion:  [B, T, J, 3] or [B, T, V, 3]
    raw_out: [B, 4]

    return: transformed motion
    """

    B = motion.shape[0]

    yaw = raw_out[:, 0]
    T = raw_out[:, 1:]
    
    # transform yaw to rotation matrix R
    R = _axis_angle_rotation("Y", yaw)

    # apply R, T to motion
    motion_flat = motion.reshape(B, -1, 3)
    motion_flat = motion_flat @ R.transpose(1, 2) + T[:, None, :]

    return motion_flat.reshape(B, *motion.shape[1:])


def place_obj_global(B, obj_center_heights, obj_scales):
    """
    Place object at origin (XZ), facing -Z.
    """

    device = obj_center_heights.device

    # translation: object placed at origin in XZ
    trans = torch.stack(
        [
            torch.zeros(B, device=device),
            obj_center_heights * obj_scales,
            torch.zeros(B, device=device),
        ],
        dim=-1
    )  # [B, 3]

    # rotation: original object faces +Z, rotate 180 deg around Y to face -Z
    yaw = torch.full(
        (B,),
        math.pi,
        device=device
    )

    rot = _axis_angle_rotation("Y", yaw)   # [B, 3, 3], (world <- obj_local)

    return trans, rot


def sample_obj_global_tr_region(B, obj_center_heights, obj_scales, region_size=3.0, yaw_noise_deg=15.0):
    """
    sample obj trans & orient in region center at (0,0,0) on XZ plane, size 3x3 m
    """
    
    device = obj_center_heights.device
    
    half = region_size / 2.0
    xs = torch.empty(B, device=device).uniform_(-half, half)
    zs = torch.empty(B, device=device).uniform_(-half, half)
    trans = torch.stack([xs, obj_center_heights * obj_scales, zs], dim=-1)  # [B, 3]

    # choose yaw pointing roughly to origin
    vx = -xs
    vz = -zs
    base_yaw = torch.atan2(vx, vz)
    yaw_noise = (torch.empty(B, device=device).uniform_(-yaw_noise_deg, yaw_noise_deg) * math.pi / 180.0)
    yaw = base_yaw + yaw_noise
    rot = _axis_angle_rotation("Y", yaw)
    
    return trans, rot
