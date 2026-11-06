import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYOPENGL_PLATFORM"] = "egl"
from visualize.vis_utils import viz_smpl_seq, create_video
import numpy as np
import trimesh
import torch
from utils.config import *


def _wrap_caption(caption, max_chars=50):
    words = str(caption).strip().split()
    if not words:
        return ""
    lines = []
    line = words[0]
    for word in words[1:]:
        candidate = f"{line} {word}"
        if len(candidate) <= max_chars:
            line = candidate
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return "\n".join(lines)

def load_obj(obj_id):
    """Load textured mesh for obj_id"""
    obj_path = os.path.join(OBJ_DIR, str(obj_id), 'transform_model.obj')
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"OBJ not found: {obj_path}")
    mesh = trimesh.load_mesh(obj_path, process=False)   
    return mesh

def load_textured_dsmal(species):
    """Load textured mesh for species"""
    key = species.replace(' ', '_')
    dsmal_path = os.path.join(DSMAL_TEX_DIR, key, 'textured.obj')
    if not os.path.exists(dsmal_path):
        raise FileNotFoundError(f"Textured DSMAL not found: {dsmal_path}")
    mesh = trimesh.load_mesh(dsmal_path, process=False, maintain_order=True)    
    return mesh

def render_body_and_object(body_verts, body_faces, species, caption, output_dir, 
                           obj_id=None, obj_trans=None, obj_orient=None, obj_scale=None, fps=30):
    # get textured dsmal
    body_verts = torch.from_numpy(body_verts) # [T, V, 3]
    body_faces = body_faces.cpu()
    texture_body_mesh = load_textured_dsmal(species)
    body_meshes = []
    for i in range(len(body_verts)):
        body_mesh = texture_body_mesh.copy()
        body_mesh.vertices = body_verts[i]
        body_meshes.append(body_mesh)

    # get textured obj
    if obj_id is not None:
        obj_mesh = load_obj(obj_id)
        S = np.eye(4, dtype=np.float32)
        S[0,0], S[1,1], S[2,2] = [float(obj_scale)] * 3
        RT = np.eye(4, dtype=np.float32)
        RT[:3, :3] = obj_orient
        RT[:3, 3] = obj_trans
        obj_mesh.apply_transform(RT @ S)
        obj_mesh = [obj_mesh]

    imw=480
    imh=480
    default_cam_pose_back = np.eye(4)
    default_cam_pose_top = trimesh.transformations.rotation_matrix(np.radians(180), (0, 0, 1)) @ \
                           trimesh.transformations.rotation_matrix(np.radians(90), (1, 0, 0))
    default_cam_pose_side1 = trimesh.transformations.rotation_matrix(np.radians(180), (0, 0, 1)) @ \
                             trimesh.transformations.rotation_matrix(np.radians(180), (1, 0, 0)) @ \
                             trimesh.transformations.rotation_matrix(np.radians(-90), (0, 1, 0))
    default_cam_pose_side2 = trimesh.transformations.rotation_matrix(np.radians(180), (0, 0, 1)) @ \
                             trimesh.transformations.rotation_matrix(np.radians(180), (1, 0, 0)) @ \
                             trimesh.transformations.rotation_matrix(np.radians(90), (0, 1, 0))
    cameras = {
        'top':{
            'camera_pose': default_cam_pose_top,
            'camera_offset': [0.0, 3.0, 0.0]
        },
        'side1':{
            'camera_pose': default_cam_pose_side1,
            'camera_offset': [2.4, 0.9, 0.0]
        },
        'side2':{
            'camera_pose': default_cam_pose_side2,
            'camera_offset': [-2.4, 0.9, 0.0]
        },
        'back':{
            'camera_pose': default_cam_pose_back,
            'camera_offset': [0.0, 0.9, 2.4]
        }
    }
  
    for view, pose in cameras.items():
        viz_smpl_seq(body_meshes, 
                    imw=imw, imh=imh, fps=fps,
                    render_body=True,
                    render_ground=True,
                    ground_plane=[0, 1, 0, 0], # Y-up
                    use_offscreen=True, # do not need GUI
                    out_path=output_dir+'_'+view,
                    static_meshes=obj_mesh if obj_id is not None else None,
                    follow_camera=True,
                    cam_offset=pose['camera_offset'],
                    cam_rot = pose['camera_pose'][:3, :3],
            )
        
    create_video(output_dir + '_side1/frame_%08d.' + '%s' % ('png'), output_dir + '_side1.mp4', fps)
    create_video(output_dir + '_side2/frame_%08d.' + '%s' % ('png'), output_dir + '_side2.mp4', fps)
    create_video(output_dir + '_top/frame_%08d.' + '%s' % ('png'), output_dir + '_top.mp4', fps)
    create_video(output_dir + '_back/frame_%08d.' + '%s' % ('png'), output_dir + '_back.mp4', fps)

    caption = _wrap_caption(caption).replace("'", r"\'").replace(':', r'\:')
    filter_complex = (
        "[0:v][1:v]hstack[top];"
        "[2:v][3:v]hstack[bottom];"
        "[top][bottom]vstack[grid];"
        "[grid]drawtext=fontfile=assets/fonts/DejaVuSans.ttf:text='{}':x=(w-text_w)/2:y=20:line_spacing=10:fontsize=28:"
        "fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=8"
    ).format(caption)
    
    os.system('ffmpeg -hide_banner -loglevel quiet -nostats -i %s -i %s -i %s -i %s -filter_complex "%s" -c:v libx264 -y %s > /dev/null 2>&1' % (
            output_dir + '_side1.mp4',
            output_dir + '_side2.mp4',
            output_dir + '_top.mp4',
            output_dir + '_back.mp4',
            filter_complex,
            output_dir + '_merge.mp4'
        ))

    # print(output_dir + '_merge.mp4')

    os.system('rm -f %s' % (output_dir + '_side1.mp4'))
    os.system('rm -f %s' % (output_dir + '_side2.mp4'))
    os.system('rm -f %s' % (output_dir + '_top.mp4'))
    os.system('rm -f %s' % (output_dir + '_back.mp4'))
    
    # remove the frames.
    os.system('rm -rf %s' % (output_dir + '_top'))
    os.system('rm -rf %s' % (output_dir + '_side1'))
    os.system('rm -rf %s' % (output_dir + '_side2'))
    os.system('rm -rf %s' % (output_dir + '_back'))



