import os
import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import TexturesVertex
import trimesh
from utils.config import *
import pickle

def make_ground_mesh(device, grid_size=10.0, y=0.0):

    ground_verts = torch.tensor([
        [-grid_size, y, -grid_size],
        [ grid_size, y, -grid_size],
        [ grid_size, y,  grid_size],
        [-grid_size, y,  grid_size],
    ], dtype=torch.float32, device=device)

    ground_faces = torch.tensor([[0,1,2],[0,2,3]], dtype=torch.int64, device=device)

    # neutral color, slight gray
    ground_colors = torch.tensor(
        [[0.6, 0.7, 0.7],
        [0.6, 0.7, 0.7],
        [0.6, 0.7, 0.7],
        [0.6, 0.7, 0.7]], 
        dtype=torch.float32, device=device
    )

    return ground_verts, ground_faces, ground_colors


class MeshBatchBuilder:
    def __init__(self, device):

        self.device = device
        self.OBJ_DIR = OBJ_DIR
        self.DSMAL_TEX_DIR = DSMAL_TEX_DIR

        # caches
        self._obj_cache = {}   # obj_id -> mesh
        self._obj_verts_color_cache = {} # obj_id -> verts_color
        self._dsmal_verts_color_cache = {} # species -> verts_color


    def _load_obj(self, obj_id):
        """Load textured mesh for obj_id, cache it."""
        if obj_id in self._obj_cache:
            return self._obj_cache[obj_id]

        obj_path = os.path.join(self.OBJ_DIR, str(obj_id), 'transform_model.obj')
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"OBJ not found: {obj_path}")
        mesh = load_objs_as_meshes([obj_path], device="cpu")
        self._obj_cache[obj_id] = mesh
        
        return mesh


    def _transform_obj_to_world(self, obj_id, obj_scale, obj_rotation, obj_translation):
        obj_mesh = self._load_obj(obj_id).to(self.device)
        obj_verts = obj_mesh.verts_list()[0] * obj_scale.to(self.device)
        obj_verts_world = obj_verts @ obj_rotation.to(self.device).transpose(0, 1) + obj_translation.to(self.device)
        return obj_mesh, obj_verts_world
    

    def _sample_obj_verts_colors(self, textured_mesh):

        texture = textured_mesh.textures
        verts_uv = texture.verts_uvs_list()[0]
        maps = texture.maps_list()[0]
        tex = maps.permute(2, 0, 1).unsqueeze(0)

        u = verts_uv[:, 0]
        v = 1.0 - verts_uv[:, 1]
        x = u * 2.0 - 1.0
        y = v * 2.0 - 1.0
        grid = torch.stack([x, y], dim=1).unsqueeze(0).unsqueeze(2)
        sampled = F.grid_sample(tex, grid, mode='bilinear', padding_mode='border', align_corners=True)
        sampled = sampled.squeeze(0).squeeze(-1).permute(1, 0)
        
        return sampled
    

    def _sample_dsmal_verts_colors(self, textured_mesh):

        tex = textured_mesh.textures
        verts = textured_mesh.verts_list()[0]
        faces = textured_mesh.faces_list()[0]
        verts_uvs = tex.verts_uvs_list()[0]
        faces_uvs = tex.faces_uvs_list()[0] # [F,3] indices into verts_uvs
        maps = tex.maps_list()[0]

        tex_img = maps.permute(2, 0, 1).unsqueeze(0)

        face_uv_coords = verts_uvs[faces_uvs]
        u = face_uv_coords[..., 0]
        v = 1.0 - face_uv_coords[..., 1]

        grid = torch.stack([u, v], dim=-1)
        Fnum = grid.shape[0]
        grid = grid.view(Fnum * 3, 2)
        grid = grid * 2.0 - 1.0
        grid = grid.view(1, Fnum * 3, 1, 2) # [1, F*3, 1, 2]

        sampled = F.grid_sample(tex_img, grid, mode='bilinear', padding_mode='border', align_corners=True)
        sampled = sampled.squeeze(-1).squeeze(0)
        sampled = sampled.permute(1,0).view(Fnum, 3, 3)  # [F, 3(verts_per_face), 3(rgb)]

        # add face-vertex color to verts
        verts_color_accum = torch.zeros((verts.shape[0], 3))
        verts_color_count = torch.zeros((verts.shape[0],))

        for i in range(3):
            idx = faces[:, i]
            verts_color_accum = verts_color_accum.index_add(0, idx, sampled[:, i, :])
            verts_color_count = verts_color_count.index_add(0, idx, torch.ones_like(idx, dtype=torch.float))

        verts_color = verts_color_accum / verts_color_count.unsqueeze(1).clamp_min(1e-6)
        
        return verts_color
           

    def _load_obj_vc(self, obj_id):
        obj_vc_path = os.path.join(self.OBJ_DIR, str(obj_id), 'transform_model.ply')
        
        if not os.path.exists(obj_vc_path):
            raise FileNotFoundError(f"OBJ Verts Color not found: {obj_vc_path}")
        
        obj_vc = trimesh.load(obj_vc_path, process=False)
        verts_color = torch.tensor(obj_vc.visual.vertex_colors[:, :3] / 255.0, dtype=torch.float32)
              
        return verts_color


    def _load_textured_dsmal_vc(self, species):
        key = species.replace(' ', '_')
        dsmal_vc_path = os.path.join(self.DSMAL_TEX_DIR, str(key), 'verts_color.pkl')
        
        if os.path.exists(dsmal_vc_path):
            with open(dsmal_vc_path, "rb") as f:
                verts_color = pickle.load(f)
        else:
            dsmal_path = os.path.join(self.DSMAL_TEX_DIR, str(key), "textured.obj")
            if not os.path.exists(dsmal_path):
                raise FileNotFoundError(f"Textured DSMAL not found: {dsmal_path}")
            mesh = load_objs_as_meshes([dsmal_path], device="cpu")
            verts_color = self._sample_dsmal_verts_colors(mesh)
            with open(dsmal_vc_path, "wb") as f:
                pickle.dump(verts_color, f)
        
        return verts_color

    
    def get_verts_color(self, key, type="obj"):
        cache = self._obj_verts_color_cache if type == "obj" else self._dsmal_verts_color_cache
        if key in cache:
            return cache[key]
        
        load_func = self._load_obj_vc if type == "obj" else self._load_textured_dsmal_vc
        verts_colors = load_func(key)   # [V,3]        
        cache[key] = verts_colors
        
        return verts_colors
        
    
    # ---------- main builder ----------
    def build_scene_meshes(self, d_verts, d_faces, y):
        """
        d_verts: torch.tensor [B, N, Vd, 3]  (D-SMAL dynamic verts)
        d_faces: torch.tensor [Fd,3]
        y: dict with keys:
            'obj_id' -> list length B
            'obj_scales' -> torch.tensor [B,]
            'obj_transform_trans' -> torch.tensor [B,3]
            'obj_transform_rotation' -> torch.tensor [B,3,3]
            'species' -> list length B
        returns:
            meshes: pytorch3d Meshes with batch size B*N, textures as TexturesVertex
        """

        B, T, Vd, _ = d_verts.shape

        ground_verts, ground_faces, ground_colors = make_ground_mesh(self.device)

        mesh_list = []
        for i in range(B):            
            # get smal & obj verts color
            smal_verts_color = self.get_verts_color(y['species'][i], type='d_smal').to(self.device)  # [Vd, 3]
            obj_verts_color = self.get_verts_color(y['obj_id'][i], type='obj').to(self.device)  # [Vo, 3]
            
            # process obj mesh
            obj_mesh, obj_verts_world = self._transform_obj_to_world(
                y['obj_id'][i],
                y['obj_scales'][i],
                y['obj_transform_rotation'][i],
                y['obj_transform_trans'][i],
            )
            
            # add faces & colors
            obj_faces_offset = obj_mesh.faces_list()[0] + Vd
            ground_faces_offset = ground_faces + Vd + obj_verts_world.shape[0]
            faces_all = torch.cat([d_faces, obj_faces_offset, ground_faces_offset], dim=0)
            faces_all = faces_all.unsqueeze(0).expand(T, -1, -1).contiguous() # [T, Fd+Fo+Fg, 3]
            
            colors_all = torch.cat([smal_verts_color, obj_verts_color, ground_colors], dim=0)
            colors_all = colors_all.unsqueeze(0).expand(T, -1, -1).contiguous() # [T, Vd+Vo+Vg, 3]
            texture_all = TexturesVertex(verts_features=colors_all)

            # add verts
            obj_verts_world = obj_verts_world.unsqueeze(0).expand(T, -1, -1).contiguous()
            ground_verts_exp = ground_verts.unsqueeze(0).expand(T, -1, -1).contiguous()
            verts_all = torch.cat([d_verts[i], obj_verts_world, ground_verts_exp], dim=1) # [T, Vd+Vo+Vg, 3]

            mesh = Meshes(verts=verts_all, faces=faces_all, textures=texture_all)
            mesh_list.append(mesh)
        
        meshes = join_meshes_as_batch(mesh_list)

        return meshes

