import os
import numpy as np
from scipy import linalg
from scipy.ndimage import uniform_filter1d
import torch
import trimesh
from utils.sdf_utils import query_feature_grid_3D
from utils.config import *
from data_loaders.tensors import lengths_to_mask
from utils.sample_frames import uniform_sample, cluster_sample
from torch.nn import functional as F
import clip

# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        correct_vec = (correct_vec | bool_mat[:, i])
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0)
    else:
        return top_k_mat


def calculate_matching_score(embedding1, embedding2, sum_all=False):
    assert len(embedding1.shape) == 2
    assert embedding1.shape[0] == embedding2.shape[0]
    assert embedding1.shape[1] == embedding2.shape[1]

    dist = linalg.norm(embedding1 - embedding2, axis=1)
    if sum_all:
        return dist.sum(axis=0)
    else:
        return dist


def calculate_activation_statistics(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative dataset set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative dataset set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        # if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
        #     m = np.max(np.abs(covmean.imag))
        #     raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_skating_ratio(motions, m_lens):
    thresh_height = 0.3
    fps = 30.0
    thresh_vel = 0.50 # 20 cm /s 
    avg_window = 5 # frames

    verts_feet = motions[:, [10, 14, 20, 24], :, :].detach().cpu().numpy()  # [bs, 4, 3, max_len]
    verts_feet_plane_vel = np.linalg.norm(verts_feet[:, :, [0, 2], 1:] - verts_feet[:, :, [0, 2], :-1],  axis=2) * fps
    vel_avg = uniform_filter1d(verts_feet_plane_vel, axis=-1, size=avg_window, mode='constant', origin=0)

    verts_feet_height = verts_feet[:, :, 1, :]
    # If feet touch ground in agjecent frames
    feet_contact = np.logical_and((verts_feet_height[:, :, :-1] < thresh_height), (verts_feet_height[:, :, 1:] < thresh_height))
    # skate velocity
    skate_vel = feet_contact * vel_avg

    # it must both skating in the current frame
    skating = np.logical_and(feet_contact, (verts_feet_plane_vel > thresh_vel))
    # and also skate in the windows of frames
    skating = np.logical_and(skating, (vel_avg > thresh_vel))

    # Both feet slide
    skating = skating.any(axis=1) # [bs, max_len -1]

    m_lens_np = m_lens.cpu().numpy().astype(int)
    T_minus_1 = verts_feet_plane_vel.shape[-1]  # max_len - 1

    # valid length for these arrays is m_lens[b] - 1
    valid_counts = np.maximum(m_lens_np - 1, 0)  # [bs]

    # build valid mask over time dim (shape [bs, T_minus_1])
    time_idx = np.arange(T_minus_1)[None, :]
    valid_time_mask = time_idx < valid_counts[:, None]  # [bs, T_minus_1], bool

    # apply mask to skating (invalidate frames >= m_lens)
    skating = skating & valid_time_mask  # [bs, T_minus_1]

    # mask skate_vel in invalid frames (expand mask to foot dim)
    valid_time_mask_exp = valid_time_mask[:, None, :]  # [bs, 1, T_minus_1]
    skate_vel = skate_vel * valid_time_mask_exp  # zero out invalid frames

    skating_ratio = np.sum(skating, axis=1) / valid_counts  # [bs]

    return skating_ratio, skate_vel
    

def calculate_pene_ratio_plane(verts, m_lens):
    
    # ground is at y=0
    verts = verts.detach().cpu().numpy()
    ground_sdf = verts[..., 1] # [B, N, V]

    B, N, V = ground_sdf.shape
    m_lens_np = m_lens.cpu().numpy().astype(int)

    # valid_frame_mask[b, t] = (t < m_lens[b])
    frame_idx = np.arange(N)[None, :]
    valid_frame_mask = frame_idx < m_lens_np[:, None] # [B, N]

    total_valid_frames = valid_frame_mask.sum(axis=1)
    total_points = total_valid_frames * V 

    pene_mask = (ground_sdf < 0.0) & valid_frame_mask[:, :, None]
    num_pen_points = pene_mask.sum(axis=(1, 2)).astype(np.int64)
    
    pen_depth = np.where(pene_mask, -ground_sdf, 0.0)
    total_depth = pen_depth.sum(axis=(1, 2))
    plane_pene_max = pen_depth.max(axis=(1, 2))

    plane_pene_mean = np.where(num_pen_points > 0, total_depth / np.maximum(num_pen_points, 1), 0.0)
    plane_pene_ratio = np.where(total_points > 0, num_pen_points / total_points, 0.0)

    return plane_pene_mean, plane_pene_ratio, plane_pene_max


def calculate_pene_ratio_obj(obj_coord_verts, obj_ids, m_lens, obj_info):

    sdf_grids, _, sdf_centers, sdf_scales = obj_info.load_batch_sdf(obj_ids, device=obj_coord_verts.device)
    sdf_values = query_feature_grid_3D(obj_coord_verts, sdf_grids, sdf_centers, sdf_scales).squeeze()
    
    sdf_values = sdf_values.view(*obj_coord_verts.shape[:-1]).cpu().numpy() # [B, N, V]
    m_lens_np = m_lens.cpu().numpy().astype(int)

    B, N, V = sdf_values.shape

    # valid_frame_mask[b, t] = (t < m_lens[b])
    frame_idx = np.arange(N)[None, :]
    valid_frame_mask = frame_idx < m_lens_np[:, None]

    total_valid_frames = valid_frame_mask.sum(axis=1)
    total_points = total_valid_frames * V 

    inside_mask = (sdf_values < -0.03) & valid_frame_mask[:, :, None] # [B, N, V], bool
    num_pen_points = inside_mask.sum(axis=(1, 2)).astype(np.int64)
    
    pen_depth = np.where(inside_mask, -sdf_values, 0.0)
    total_depth = pen_depth.sum(axis=(1, 2))
    obj_pene_max = pen_depth.max(axis=(1, 2))

    obj_pene_mean = np.where(num_pen_points > 0, total_depth / np.maximum(num_pen_points, 1), 0.0)
    obj_pene_ratio = np.where(total_points > 0, num_pen_points / total_points, 0.0)

    return obj_pene_mean, obj_pene_ratio, obj_pene_max


def calculate_motion_diversity(motions, m_lens):
    K = motions.shape[0]
    dists = []
    for i in range(K):
        for j in range(i + 1, K):
            T = min(m_lens[i], m_lens[j])
            diff = motions[i, :T] - motions[j, :T]
            dist = torch.norm(diff, dim=-1)   # [T, J]
            dists.append(dist.mean())
    return torch.stack(dists).mean()


CLIP_MEAN = torch.tensor(
    (0.48145466, 0.4578275, 0.40821073)
).view(1, 3, 1, 1)

CLIP_STD = torch.tensor(
    (0.26862954, 0.26130258, 0.27577711),
).view(1, 3, 1, 1)

def encode_text(raw_text, clip_model, device):
    texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
    texts_feat = clip_model.encode_text(texts).float()
    return F.normalize(texts_feat, dim=-1)  # [B, C]

def encode_image(images, clip_model, device):
    images = images.permute(0, 3, 1, 2)  # [B*k, 3, H, W], range [0,1]
    images = (images - CLIP_MEAN.to(device)) / CLIP_STD.to(device)  # normalize, range [-1,1]
    images_feat = clip_model.encode_image(images).float()
    return F.normalize(images_feat, dim=-1)  # [B*k, C]

def evaluate_clip(verts, joints, captions, species, obj_ids, lengths, 
                  obj_transform_trans, obj_transform_rotations, obj_scales, 
                  mesh_builder, eval_renderer, smal_model, clip_model, sample_frame=20):
    
    batch_size = verts.shape[0]  # Store original batch size
    device = verts.device
    
    # sample keyframes
    length_mask = lengths_to_mask(lengths, verts.shape[1]).to(device)
    keyframe_idx = uniform_sample(length_mask, k=sample_frame) # [B, k]
    
    batch_idx = torch.arange(verts.shape[0], device=device)[:, None]
    smal_verts_sampled = verts[batch_idx, keyframe_idx] # [B, k, V, 3]
    root_pos_sampled = joints[batch_idx, keyframe_idx, 0].reshape(-1, 3) # [B*k, 3]

    # get smal+obj mesh in WCS
    y = {
        'obj_id': obj_ids,
        'obj_scales': obj_scales,
        'obj_transform_trans': obj_transform_trans,
        'obj_transform_rotation': obj_transform_rotations,
        'species': species,
    }
    # build meshes and render video frames
    meshes = mesh_builder.build_scene_meshes(smal_verts_sampled, smal_model.faces.to(device), y)
    images = eval_renderer.render_meshes_partition(meshes, root_pos_sampled, chunksize=64) # [B*k, H, W, 3]
    
    # encode with CLIP
    with torch.no_grad():
        clip_image_feat = encode_image(images, clip_model, device).reshape(batch_size, sample_frame, -1)  # [B, k, C]    
        clip_text_feat = encode_text(captions, clip_model, device)[:, None, :]  # [B, 1, C]

    # --------------------------------------------------
    # (1) CLIP Score (Image–Text)
    # --------------------------------------------------
    # cosine similarity since both are normalized
    clip_sim = (clip_image_feat * clip_text_feat).sum(dim=-1)
    clip_score = clip_sim.mean(dim=1)

    # --------------------------------------------------
    # (2) Frame-wise CLIP Consistency
    # --------------------------------------------------
    img_feat_t = clip_image_feat[:, :-1, :]
    img_feat_t1 = clip_image_feat[:, 1:, :]

    consistency = (img_feat_t * img_feat_t1).sum(dim=-1)
    clip_consistency = consistency.mean(dim=1)

    return clip_score * 100.0, clip_consistency


  
