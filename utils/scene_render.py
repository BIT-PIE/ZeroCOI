import torch
from torch import nn
import numpy as np
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,
    HardPhongShader,
    PointLights,
    BlendParams,
    look_at_view_transform
)

def partition_index(totalsize: int, chunksize: int) -> list[tuple[int, int]]:
    a, b = 0, 0
    if totalsize <= chunksize:
        return [(a, totalsize)]
    list_tuples = []
    b = chunksize
    while b < totalsize:
        list_tuples.append((a, b))
        a = b
        b += chunksize
    list_tuples.append((a, min(totalsize, b)))
    return list_tuples


class SceneRenderer(nn.Module):
    def __init__(self, camera_mode='random', image_size=(160, 256), faces_per_pixels=1, device='cuda', fov=80.0):
        """
        Pytorch3d-based renderer, use for SDS
        """
        super(SceneRenderer, self).__init__()

        self.device = device
        self.image_size = image_size
        self.fov = fov
        self.camera_mode = camera_mode

        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=faces_per_pixels,
            max_faces_per_bin=50000,
        )

        # create a default camera & lights to initialize renderer once
        dummy_R = torch.eye(3, device=self.device).unsqueeze(0)
        dummy_T = torch.zeros((1,3), device=self.device)
        dummy_cameras = FoVPerspectiveCameras(device=self.device, R=dummy_R, T=dummy_T, fov=self.fov)
        dummy_lights = PointLights(device=self.device, location=torch.zeros((1,3), device=self.device))

        # create rasterizer + shader + renderer once
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=dummy_cameras, raster_settings=self.raster_settings),
            shader=HardPhongShader(device=self.device, cameras=dummy_cameras, lights=dummy_lights, blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)),) # white background
        )

    def sample_random_view(self, 
                    left_az_range: tuple = (-160.0, -110.0),
                    right_az_range: tuple = (110.0, 160.0),
                    elevation_range: tuple = (10.0, 60.0),
                    dist_range: tuple = (1.3, 2.0),
                ):
        
        # sample a single random spherical offset (shared by all frames in batch)
        side = 'left' if torch.rand(1).item() < 0.5 else 'right'
        azimuth_range = left_az_range if side == 'left' else right_az_range
        az = float(torch.empty(1).uniform_(azimuth_range[0], azimuth_range[1]).item())
        el = float(torch.empty(1).uniform_(elevation_range[0], elevation_range[1]).item())

        # sample distance uniformly in dist_range
        dist = float(torch.empty(1).uniform_(dist_range[0], dist_range[1]).item())

        return az, el, dist
    
    def get_camera_lights(self, root_pos: torch.Tensor):

        az, el, dist = self.sample_random_view() if self.camera_mode == 'random' else (120.0, 50.0, 1.7)
        
        az_rad = torch.deg2rad(torch.tensor(az, device=self.device))
        el_rad = torch.deg2rad(torch.tensor(el, device=self.device))

        x = torch.cos(el_rad) * torch.sin(az_rad)
        y = torch.sin(el_rad)
        z = torch.cos(el_rad) * torch.cos(az_rad)
        direction = torch.stack([x, y, z], dim=0)

        offset = direction * dist

        cam_eye = root_pos + offset.unsqueeze(0) # apply the same offset to every root_pos to get camera eyes
        look_at = root_pos.clone() # each camera looks at the root

        R, T = look_at_view_transform(
            eye=cam_eye,
            at=look_at,
            up=torch.tensor([[0.0, 1.0, 0.0]], device=self.device, dtype=torch.float32)
        )

        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=self.fov)

        return cameras, cam_eye


    def render_meshes_partition(self, meshes, root_pos, chunksize=256):

        cameras, cam_eye = self.get_camera_lights(root_pos.detach())

        indexes_chunk = partition_index(totalsize=len(meshes), chunksize=chunksize)
        
        return torch.cat([self.render_meshes(meshes[a:b], cameras[list(range(a, b))], cam_eye[a:b]) for (a, b) in indexes_chunk], dim=0)
      

    def render_meshes(self, meshes, cameras, cam_eye):

        self.renderer.rasterizer.cameras = cameras
        self.renderer.shader.cameras = cameras
        
        self.renderer.shader.lights.location = cam_eye
        
        images = self.renderer(meshes)[..., :3].clamp(0.0, 1.0)
        
        return images
