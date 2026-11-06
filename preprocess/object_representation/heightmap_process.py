import argparse
import numpy as np
from scipy.ndimage import gaussian_filter
import trimesh
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from tqdm import tqdm

def compute_heightmap_from_mesh_rasterization(
    mesh_vertices,
    mesh_faces, 
    resolution=128, 
    smooth_sigma=0.5,
    margin_ratio=0.00,
    upward_threshold=0.0
):
    """
    Compute a precise heightmap by rasterizing mesh triangles.
    
    Args:
        mesh_vertices: [V, 3] numpy array of mesh vertices (x, y, z).
        mesh_faces: [F, 3] numpy array of triangle indices.
        resolution: Heightmap resolution.
        smooth_sigma: Gaussian smoothing sigma.
        margin_ratio: Expansion ratio for xz bounds.
        upward_threshold: Upward normal threshold on y component.
    
    Returns:
        dict: {
            'heightmap': [resolution, resolution],
            'support_height': scalar,
            'xz_min': [2,],
            'xz_max': [2,],
            'resolution': int
        }
    """
    v0 = mesh_vertices[mesh_faces[:, 0]]  # [F, 3]
    v1 = mesh_vertices[mesh_faces[:, 1]]
    v2 = mesh_vertices[mesh_faces[:, 2]]
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    normals = np.cross(edge1, edge2)  # [F, 3]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norms + 1e-8)
    
    # Keep upward-facing triangles only.
    upward_mask = normals[:, 1] > upward_threshold
    upward_faces = mesh_faces[upward_mask]
    
    if len(upward_faces) == 0:
        # Fallback: use all faces if no upward faces are found.
        upward_faces = mesh_faces
    
    # Build xz bounds from upward faces.
    upward_vertices_idx = np.unique(upward_faces.flatten())
    upward_vertices = mesh_vertices[upward_vertices_idx]
    xz_upward = upward_vertices[:, [0, 2]]
    xz_min_full = xz_upward.min(axis=0)
    xz_max_full = xz_upward.max(axis=0)
    xz_range = xz_max_full - xz_min_full
    xz_min_full -= xz_range * margin_ratio
    xz_max_full += xz_range * margin_ratio
    
    heightmap = np.full((resolution, resolution), -np.inf)
    
    # Query points on xz grid.
    x_grid = np.linspace(xz_min_full[0], xz_max_full[0], resolution)
    z_grid = np.linspace(xz_min_full[1], xz_max_full[1], resolution)
    grid_x, grid_z = np.meshgrid(x_grid, z_grid)  # [res, res]
    query_points = np.stack([grid_x.ravel(), grid_z.ravel()], axis=1)  # [res*res, 2]
    
    # Rasterize each upward triangle onto candidate grid points.
    for face_idx in upward_faces:
        tri_vertices = mesh_vertices[face_idx]  # [3, 3]
        tri_xz = tri_vertices[:, [0, 2]]  # [3, 2]
        tri_y = tri_vertices[:, 1]  # [3]
        
        # Triangle bounding box in xz.
        tri_xz_min = tri_xz.min(axis=0)
        tri_xz_max = tri_xz.max(axis=0)
        
        # Select candidate points within the bounding box.
        in_bbox_x = (query_points[:, 0] >= tri_xz_min[0]) & (query_points[:, 0] <= tri_xz_max[0])
        in_bbox_z = (query_points[:, 1] >= tri_xz_min[1]) & (query_points[:, 1] <= tri_xz_max[1])
        candidate_mask = in_bbox_x & in_bbox_z
        
        if not candidate_mask.any():
            continue
        
        candidate_points = query_points[candidate_mask]  # [N, 2]
        
        # Barycentric test + y interpolation.
        v0_xz = tri_xz[0]
        v1_xz = tri_xz[1]
        v2_xz = tri_xz[2]
        
        v0v1 = v1_xz - v0_xz
        v0v2 = v2_xz - v0_xz
        v0p = candidate_points - v0_xz[None, :]  # [N, 2]
        
        # 2D cross product.
        def cross2d(a, b):
            return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        
        denom = cross2d(v0v1[None, :].repeat(len(v0p), axis=0), v0v2[None, :].repeat(len(v0p), axis=0))
        u = cross2d(v0p, v0v2[None, :].repeat(len(v0p), axis=0)) / (denom + 1e-8)
        v = cross2d(v0v1[None, :].repeat(len(v0p), axis=0), v0p) / (denom + 1e-8)
        
        # Inside-triangle condition: u>=0, v>=0, u+v<=1.
        inside_mask = (u >= 0) & (v >= 0) & (u + v <= 1)
        
        if not inside_mask.any():
            continue
        
        # Interpolate y.
        u_inside = u[inside_mask]
        v_inside = v[inside_mask]
        w = 1 - u_inside - v_inside
        y_interp = w * tri_y[0] + u_inside * tri_y[1] + v_inside * tri_y[2]
        
        # Update heightmap with max y.
        candidate_indices = np.where(candidate_mask)[0][inside_mask]
        grid_i = candidate_indices // resolution
        grid_j = candidate_indices % resolution
        
        for idx, (i, j) in enumerate(zip(grid_i, grid_j)):
            heightmap[i, j] = max(heightmap[i, j], y_interp[idx])
    
    # Light smoothing on valid cells.
    if smooth_sigma > 0:
        valid_mask = (heightmap > -np.inf)
        if valid_mask.any():
            temp_heightmap = heightmap.copy()
            mean_val = heightmap[valid_mask].mean()
            temp_heightmap[~valid_mask] = mean_val
            smoothed = gaussian_filter(temp_heightmap, sigma=smooth_sigma)
            heightmap[valid_mask] = smoothed[valid_mask]
    
    # Fill empty cells with mesh minimum y.
    empty_mask = (heightmap == -np.inf)
    if empty_mask.any():
        min_height = float(mesh_vertices[:, 1].min())
        heightmap[empty_mask] = min_height
    
    xz_min = xz_min_full.astype(np.float32)
    xz_max = xz_max_full.astype(np.float32)
    
    # Support height as median of valid heights.
    valid_heights = heightmap[heightmap > -np.inf]
    if len(valid_heights) > 0:
        support_height = float(np.median(valid_heights))
    else:
        support_height = 0.0
    
    return {
        'heightmap': heightmap.astype(np.float32),
        'support_height': support_height,
        'xz_min': xz_min,
        'xz_max': xz_max,
        'resolution': resolution
    }


def visualize_heightmap(heightmap_data, mesh_vertices, mesh_faces, output_path):
    """
    Visualize heightmap as 2D heatmap and 3D overlay.
    
    Args:
        heightmap_data: Heightmap dictionary.
        mesh_vertices: Mesh vertices.
        mesh_faces: Mesh faces.
        output_path: Output image path.
    """
    fig = plt.figure(figsize=(12, 5))
    
    # 2D heatmap.
    ax1 = fig.add_subplot(121)
    im = ax1.imshow(heightmap_data['heightmap'], origin='lower', cmap='viridis', aspect='auto')
    ax1.set_title(f'Heightmap (2D)\nSupport Height: {heightmap_data["support_height"]:.3f}m')
    ax1.set_xlabel('X grid')
    ax1.set_ylabel('Z grid')
    plt.colorbar(im, ax=ax1, label='Height (m)')
    
    # 3D mesh + heightmap overlay + xz box.
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_trisurf(mesh_vertices[:, 0], mesh_vertices[:, 2], mesh_vertices[:, 1],
                     triangles=mesh_faces, alpha=0.15, edgecolor='gray', linewidth=0.1, color='cyan')
    
    # Heightmap surface.
    x_grid = np.linspace(heightmap_data['xz_min'][0], heightmap_data['xz_max'][0], heightmap_data['resolution'])
    z_grid = np.linspace(heightmap_data['xz_min'][1], heightmap_data['xz_max'][1], heightmap_data['resolution'])
    X, Z = np.meshgrid(x_grid, z_grid)
    Y = heightmap_data['heightmap']
    ax2.plot_surface(X, Z, Y, cmap='hot', alpha=0.6, edgecolor='none')
    
    # xz boundary box.
    xz_min = heightmap_data['xz_min']
    xz_max = heightmap_data['xz_max']
    y_min = heightmap_data['heightmap'].min()
    y_max = heightmap_data['heightmap'].max()
    
    # Bottom edges.
    ax2.plot([xz_min[0], xz_max[0]], [xz_min[1], xz_min[1]], [y_min, y_min], 'r-', linewidth=2, label='XZ Boundary')
    ax2.plot([xz_max[0], xz_max[0]], [xz_min[1], xz_max[1]], [y_min, y_min], 'r-', linewidth=2)
    ax2.plot([xz_max[0], xz_min[0]], [xz_max[1], xz_max[1]], [y_min, y_min], 'r-', linewidth=2)
    ax2.plot([xz_min[0], xz_min[0]], [xz_max[1], xz_min[1]], [y_min, y_min], 'r-', linewidth=2)
    
    # Top edges.
    ax2.plot([xz_min[0], xz_max[0]], [xz_min[1], xz_min[1]], [y_max, y_max], 'r-', linewidth=2)
    ax2.plot([xz_max[0], xz_max[0]], [xz_min[1], xz_max[1]], [y_max, y_max], 'r-', linewidth=2)
    ax2.plot([xz_max[0], xz_min[0]], [xz_max[1], xz_max[1]], [y_max, y_max], 'r-', linewidth=2)
    ax2.plot([xz_min[0], xz_min[0]], [xz_max[1], xz_min[1]], [y_max, y_max], 'r-', linewidth=2)
    
    # Vertical edges.
    ax2.plot([xz_min[0], xz_min[0]], [xz_min[1], xz_min[1]], [y_min, y_max], 'r-', linewidth=2)
    ax2.plot([xz_max[0], xz_max[0]], [xz_min[1], xz_min[1]], [y_min, y_max], 'r-', linewidth=2)
    ax2.plot([xz_max[0], xz_max[0]], [xz_max[1], xz_max[1]], [y_min, y_max], 'r-', linewidth=2)
    ax2.plot([xz_min[0], xz_min[0]], [xz_max[1], xz_max[1]], [y_min, y_max], 'r-', linewidth=2)
    
    ax2.set_title('Mesh + Heightmap + XZ Boundary')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_zlabel('Height (m)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_single_object(obj_folder, resolution=64, smooth_sigma=0.5, upward_threshold=0.0):
    """
    Process one object: compute and save heightmap.
    
    Args:
        obj_folder: Object folder path.
        resolution: Heightmap resolution.
        smooth_sigma: Smoothing sigma.
        upward_threshold: Upward face threshold.
    
    Returns:
        bool: Whether processing succeeds.
    """
    obj_folder = Path(obj_folder)
    mesh_path = obj_folder / "transform_model.obj"
    
    if not mesh_path.exists():
        print(f"  Skip: {obj_folder.name} (missing transform_model.obj)")
        return False
    
    try:
        mesh = trimesh.load(str(mesh_path))
        mesh_vertices = np.array(mesh.vertices)
        mesh_faces = np.array(mesh.faces)
        
        heightmap_data = compute_heightmap_from_mesh_rasterization(
            mesh_vertices,
            mesh_faces,
            resolution=resolution,
            smooth_sigma=smooth_sigma,
            margin_ratio=0.0,
            upward_threshold=upward_threshold
        )
        
        pkl_path = obj_folder / "height.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(heightmap_data, f)
        
        vis_path = obj_folder / "heightmap_vis.png"
        visualize_heightmap(heightmap_data, mesh_vertices, mesh_faces, str(vis_path))
        
        return True
        
    except Exception as e:
        print(f"  Error: {obj_folder.name} - {str(e)}")
        return False


def batch_process(dataset_root, resolution=64, smooth_sigma=0.5, upward_threshold=0.0):
    """
    Batch process all objects.
    
    Args:
        dataset_root: Dataset root directory.
        resolution: Heightmap resolution.
        smooth_sigma: Smoothing sigma.
        upward_threshold: Upward face threshold.
    """
    dataset_root = Path(dataset_root)
    obj_folders = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    
    print(f"Found {len(obj_folders)} object folders")
    print(f"Params: resolution={resolution}, smooth_sigma={smooth_sigma}, upward_threshold={upward_threshold}")
    print("="*80)
    
    success_count = 0
    
    for obj_folder in tqdm(obj_folders, desc="Processing"):
        if process_single_object(obj_folder, resolution, smooth_sigma, upward_threshold):
            success_count += 1
    
    print("="*80)
    print(f"Done! Processed {success_count}/{len(obj_folders)} objects")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heightmap preprocessing")
    parser.add_argument("--root_dir", "--base_dir", dest="root_dir", type=Path, required=True)
    args = parser.parse_args()

    batch_process(
        args.root_dir,
        resolution=64, 
        smooth_sigma=0.5, 
        upward_threshold=0.0
    )
