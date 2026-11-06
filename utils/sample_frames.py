import torch

def uniform_sample(valid_mask, k=16):
    """
    Uniformly sample k frames from valid frames for each sequence.
    
    Args:
        valid_mask: [B, N] binary mask indicating valid frames
        k: number of frames to sample
    
    Returns:
        indices: [B, k] sampled frame indices for each sequence
    """
    B, N = valid_mask.shape
    device = valid_mask.device
    
    valid_counts = valid_mask.sum(dim=1)
    # Find start indices of valid frames
    cumsum = valid_mask.cumsum(dim=1)
    start_indices = (cumsum == 1).float().argmax(dim=1)
    
    # Generate uniform sampling positions
    uniform_positions = torch.linspace(0, 1 - 1e-6, k, device=device).unsqueeze(0).expand(B, -1)
    scaled_positions = uniform_positions * (valid_counts.unsqueeze(1) - 1).clamp(min=0)
    sampled_indices = (start_indices.unsqueeze(1) + scaled_positions).long()
        
    return sampled_indices.clamp(0, N - 1)


def cluster_sample(verts, valid_mask, k=16):
    """
    Cluster sample k frames from valid frames using K-means clustering.
    
    Args:
        verts: [B, N, V, 3] vertex positions
        valid_mask: [B, N] binary mask indicating valid frames
        k: number of frames to sample
    
    Returns:
        indices: [B, k] sampled frame indices for each sequence
    """
    from sklearn.cluster import KMeans
    
    B = verts.shape[0]
    device = verts.device
    indices = torch.zeros(B, k, dtype=torch.long, device=device)
    
    verts_for_clustering = verts.detach()
    
    for b in range(B):
        valid_mask = valid_mask[b].bool()
        valid_verts = verts_for_clustering[b][valid_mask]  # [T_valid, V, 3]
        T_valid = valid_verts.shape[0]
        
        if T_valid <= k:
            # Not enough valid frames, pad with the last frame
            valid_idx = torch.where(valid_mask)[0]
            indices[b, :T_valid] = valid_idx
            indices[b, T_valid:] = valid_idx[-1]
            continue
        
        # Use mean vertex position as embedding for clustering
        emb = valid_verts.mean(dim=1).cpu().numpy()
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(emb)
        
        # Select the frame closest to each cluster center
        selected = []
        for ci in range(k):
            members = torch.where(torch.tensor(kmeans.labels_) == ci)[0]
            center = torch.tensor(kmeans.cluster_centers_[ci], device=device)
            dists = ((valid_verts[members].mean(dim=1) - center) ** 2).sum(dim=1)
            selected.append(members[dists.argmin()].item())
        
        # Map back to original indices and sort by time
        valid_idx = torch.where(valid_mask)[0]
        indices[b] = valid_idx[torch.tensor(sorted(selected), device=device)]
    
    return indices