import time, os, shutil
import subprocess
import glob

import numpy as np
import trimesh
import torch
from PIL import Image, ImageDraw, ImageFont

from .mesh_viewer import MeshViewer, colors

copy2cpu = lambda tensor: tensor.detach().cpu().numpy()

def viz_smpl_seq(body, imw=1080, imh=1080, fps=30, contacts=None,
                render_body=True, body_color=None, render_joints=False, render_skeleton=False, render_ground=True, ground_plane=None,
                use_offscreen=False, out_path=None, wireframe=False, RGBA=False,
                joints_seq=None, joints_vel=None, follow_camera=False, vtx_list=None, points_seq=None, points_vel=None,
                static_meshes=None, camera_intrinsics=None, img_seq=None, point_rad=0.015,
                skel_connections=None, img_extn='png', ground_alpha=1.0, body_alpha=None, mask_seq=None,
                cam_offset=[0.0, 4.0, 1.25], ground_color0=[0.8, 0.9, 0.9], ground_color1=[0.6, 0.7, 0.7],
                skel_color=[0.0, 0.0, 1.0],
                joint_rad=0.015,
                point_color=[0.0, 1.0, 0.0],
                joint_color=[0.0, 1.0, 0.0],
                contact_color=[1.0, 0.0, 0.0],
                render_bodies_static=None,
                render_points_static=None,
                cam_rot=None):
    '''
    Visualizes the body model output of a smpl sequence.
    - body : body model output from SMPL forward pass (where the sequence is the batch)
    - joints_seq : list of torch/numy tensors/arrays
    - points_seq : list of torch/numpy tensors
    - camera_intrinsics : (fx, fy, cx, cy)
    - ground_plane : [a, b, c, d]
    - render_bodies_static is an integer, if given renders all bodies at once but only every x steps
    '''
    
    # import pdb;pdb.set_trace()
    if contacts is not None and torch.is_tensor(contacts):
        contacts = copy2cpu(contacts)

    # import pdb;pdb.set_trace()
    if render_body or vtx_list is not None:
        if isinstance(body, list) and isinstance(body[0], trimesh.Trimesh):
            body_mesh_seq = body

        else: # smpl object
            start_t = time.time()
            nv = body.v.size(1)
            vertex_colors = np.tile(colors[body_color], (nv, 1))
            if body_alpha is not None:
                vtx_alpha = np.ones((vertex_colors.shape[0], 1))*body_alpha
                vertex_colors = np.concatenate([vertex_colors, vtx_alpha], axis=1)
            faces = copy2cpu(body.f)
            body_mesh_seq = [trimesh.Trimesh(vertices=copy2cpu(body.v[i]), faces=faces, vertex_colors=vertex_colors, process=False) for i in range(body.v.size(0))]

    if render_joints and joints_seq is None:
        start_t = time.time()
        joints_seq = [copy2cpu(body.Jtr[i, :22]) for i in range(body.Jtr.size(0))]
    elif render_joints and torch.is_tensor(joints_seq[0]):
        joints_seq = [copy2cpu(joint_frame) for joint_frame in joints_seq]

    if joints_vel is not None and torch.is_tensor(joints_vel[0]):
        joints_vel = [copy2cpu(joint_frame) for joint_frame in joints_vel]
    if points_vel is not None and torch.is_tensor(points_vel[0]):
        points_vel = [copy2cpu(joint_frame) for joint_frame in points_vel]

    mv = MeshViewer(width=imw, height=imh,
                    use_offscreen=use_offscreen, 
                    follow_camera=follow_camera,
                    camera_intrinsics=camera_intrinsics,
                    img_extn=img_extn,
                    default_cam_offset=cam_offset,
                    default_cam_rot=cam_rot)

    if render_body and render_bodies_static is None:
        mv.add_mesh_seq(body_mesh_seq, color_name=body_color)

    elif render_body and render_bodies_static is not None:
        mv.add_static_meshes([body_mesh_seq[i] for i in range(len(body_mesh_seq)) if i % render_bodies_static == 0])
    
    # import pdb;pdb.set_trace()
    if render_joints and render_skeleton:
        mv.add_point_seq(joints_seq, color=joint_color, radius=joint_rad, contact_seq=contacts, 
                         connections=skel_connections, connect_color=skel_color, vel=joints_vel,
                         contact_color=contact_color, render_static=render_points_static)
    elif render_joints:
        mv.add_point_seq(joints_seq, color=joint_color, radius=joint_rad, contact_seq=contacts, vel=joints_vel, contact_color=contact_color,
                            render_static=render_points_static)

    if vtx_list is not None:
        mv.add_smpl_vtx_list_seq(body_mesh_seq, vtx_list, color=[0.0, 0.0, 1.0], radius=0.015)

    # import pdb;pdb.set_trace()
    if points_seq is not None:
        if torch.is_tensor(points_seq[0]):
            points_seq = [copy2cpu(point_frame) for point_frame in points_seq]
        mv.add_point_seq(points_seq, color=point_color, radius=point_rad, vel=points_vel, render_static=1)

    if static_meshes is not None:
        mv.set_static_meshes(static_meshes)

    if img_seq is not None:
        mv.set_img_seq(img_seq)

    if mask_seq is not None:
        mv.set_mask_seq(mask_seq)

    if render_ground:
        xyz_orig = None
        if ground_plane is not None:
            if render_body:
                xyz_orig = body_mesh_seq[0].vertices[0, :]
            elif render_joints:
                xyz_orig = joints_seq[0][0, :]
            elif points_seq is not None:
                xyz_orig = points_seq[0][0, :]

        mv.add_ground(ground_plane=ground_plane, xyz_orig=xyz_orig, color0=ground_color0, color1=ground_color1, alpha=ground_alpha)

    mv.set_render_settings(out_path=out_path, wireframe=wireframe, RGBA=RGBA,
                            single_frame=(render_points_static is not None or render_bodies_static is not None)) # only does anything for offscreen rendering
    try:
        mv.animate(fps=fps)
    except RuntimeError as err:
        print('Could not render properly with the error: %s' % (str(err)))

    del mv


def create_video(img_path, out_path, fps):
    '''
    Creates a video from the frame format in the given directory and saves to out_path.
    '''
    command = ['ffmpeg', '-y', '-r', str(fps), '-i', img_path, \
                    '-vcodec', 'libx264', '-crf', '25', '-pix_fmt', 'yuv420p', out_path]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)