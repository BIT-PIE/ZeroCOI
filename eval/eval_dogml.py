from utils.parser_utils import evaluation_parser
from utils.fixseed import fixseed
from datetime import datetime
from data_loaders.motion_loaders.model_motion_loaders import get_mdm_loader
from eval.metrics import *
from data_loaders.networks.evaluator_wrapper import EvaluatorMDMWrapper
from collections import OrderedDict
from utils.model_utils import create_model_and_diffusion, load_saved_model

from diffusion import logger
from utils import dist_utils
from data_loaders.get_data import get_dataset_loader
from model.cfg_sampler import ClassifierFreeSampleModel
from utils.BITE_adapter import SMAL
from utils.BITE_utils import get_verts_from_data
from utils.sdf_utils import OBJInfo
from utils.decouple_utils import apply_rt_to_motion
import clip
from utils.scene_render import SceneRenderer
from utils.mesh_builder import MeshBatchBuilder
from utils.bps_utils import canonicalize_poses_to_object_space

torch.multiprocessing.set_sharing_strategy('file_system')

def evaluate_matching_score(eval_wrapper, motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                word_embeddings, pos_one_hots, _, sent_lens, motions, _, m_lens, *_ = batch

                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens
                )
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_physic(motion_loaders, smal_model, file):
    skating_ratio_dict = OrderedDict({})
    pene_plane_ratio_dict = OrderedDict({})
    pene_plane_mean_depth_dict = OrderedDict({})
    pene_plane_max_depth_dict = OrderedDict({})

    motion_loader_name = 'vald'
    motion_loader = motion_loaders[motion_loader_name]
    print('========== Evaluating Physic ==========')
    # all_dist = []
    all_size = 0
    skate_ratio_sum = 0
    pene_plane_ratio_sum = 0
    pene_plane_depth_sum = 0
    pene_plane_max_sum = 0
    with torch.no_grad():
        for idx, batch in enumerate(motion_loader):
            word_embeddings, pos_one_hots, _, sent_lens, motions, scale_betas, m_lens, raw_out, *_ = batch
            
            # denorm motion
            mean = motion_loader.dataset.dataloader.dataset.mean
            std = motion_loader.dataset.dataloader.dataset.std
            motions = motions * std + mean
            motions = motions.float()  # (B, N, D)

            motions = motions.to(dist_utils.dev())
            scale_betas = scale_betas.to(dist_utils.dev())
            raw_out = raw_out.to(dist_utils.dev())
            verts, joints, _ = get_verts_from_data(motions, scale_betas, smal_model) # (B, N, V, 3) (B, N, J, 3)
            verts = apply_rt_to_motion(verts, raw_out)
            joints = apply_rt_to_motion(joints, raw_out)
            
            # foot skating error
            skate_ratio, skate_vel = calculate_skating_ratio(joints.permute(0, 2, 3, 1), m_lens)  # [batch_size]
            skate_ratio_sum += skate_ratio.sum()
            
            # pene plane error
            pene_plane_mean, pene_plane_ratio, pene_plane_max = calculate_pene_ratio_plane(verts, m_lens)
            pene_plane_ratio_sum += pene_plane_ratio.sum()
            pene_plane_depth_sum += pene_plane_mean.sum()
            pene_plane_max_sum += pene_plane_max.sum()
            all_size += motions.shape[0]

        skating_score = skate_ratio_sum / all_size
        skating_ratio_dict[motion_loader_name] = skating_score

        pene_ratio_score = pene_plane_ratio_sum / all_size
        pene_plane_ratio_dict[motion_loader_name] = pene_ratio_score

        pene_mean_depth_score = pene_plane_depth_sum / all_size
        pene_plane_mean_depth_dict[motion_loader_name] = pene_mean_depth_score

    print(f'---> [{motion_loader_name}] Skating Ratio: {skating_score:.4f}')
    print(f'---> [{motion_loader_name}] Skating Ratio: {skating_score:.4f}', file=file, flush=True)
    
    print(f'---> [{motion_loader_name}] Pene Ratio: {pene_ratio_score:.4f}')
    print(f'---> [{motion_loader_name}] Pene Ratio: {pene_ratio_score:.4f}', file=file, flush=True)
    print(f'---> [{motion_loader_name}] Pene Mean Depth(m): {pene_mean_depth_score:.4f}')
    print(f'---> [{motion_loader_name}] Pene Mean Depth(m): {pene_mean_depth_score:.4f}', file=file, flush=True)
    print(f'---> [{motion_loader_name}] Pene Max Depth(m): {pene_plane_max_sum / all_size:.4f}')
    print(f'---> [{motion_loader_name}] Pene Max Depth(m): {pene_plane_max_sum / all_size:.4f}', file=file, flush=True)
    
    return skating_ratio_dict, pene_plane_ratio_dict, pene_plane_mean_depth_dict, pene_plane_max_depth_dict

def evaluate_obj_interaction(motion_loaders, smal_model, obj_info, mesh_builder, eval_renderer, clip_model, file):
    
    obj_pene_ratio_dict = OrderedDict({})
    obj_pene_mean_depth_dict = OrderedDict({})
    obj_pene_max_depth_dict = OrderedDict({})

    clip_score_dict = OrderedDict({})
    clip_consistency_dict = OrderedDict({})

    motion_loader_name = 'vald'
    motion_loader = motion_loaders[motion_loader_name]
    print('========== Evaluating OBJ Interaction ==========')
    all_size = 0
    obj_pene_ratio_sum = 0
    obj_pene_depth_sum = 0
    obj_pene_max_sum = 0
    clip_score_sum = 0
    clip_consistency_sum = 0

    with torch.no_grad():
        for idx, batch in enumerate(motion_loader):
            word_embeddings, pos_one_hots, captions, sent_lens, motions, scale_betas, m_lens, raw_out, \
            obj_ids, obj_transform_trans, obj_transform_rotations, obj_scales, species = batch
            
            # denorm motion
            mean = motion_loader.dataset.dataloader.dataset.mean
            std = motion_loader.dataset.dataloader.dataset.std
            motions = motions * std + mean
            motions = motions.float()  # (B, N, D)

            motions = motions.to(dist_utils.dev())
            scale_betas = scale_betas.to(dist_utils.dev())
            raw_out = raw_out.to(dist_utils.dev())
            
            verts, joints, _ = get_verts_from_data(motions, scale_betas, smal_model) # [B, N, V, 3] 
            verts = apply_rt_to_motion(verts, raw_out)
            joints = apply_rt_to_motion(joints, raw_out)
            
            obj_transform_trans = obj_transform_trans.to(dist_utils.dev())
            obj_transform_rotations = obj_transform_rotations.to(dist_utils.dev())
            obj_scales = obj_scales.to(dist_utils.dev())
            obj_coord_verts = canonicalize_poses_to_object_space(verts, obj_transform_trans, obj_transform_rotations, obj_scales)
            
            # pene obj error
            obj_pene_mean, obj_pene_ratio, obj_pene_max = calculate_pene_ratio_obj(obj_coord_verts, obj_ids, m_lens, obj_info)
            obj_pene_ratio_sum += obj_pene_ratio.sum()
            obj_pene_depth_sum += obj_pene_mean.sum()
            obj_pene_max_sum += obj_pene_max.sum()
            
            # clip score and consistency
            clip_score, clip_consistency = evaluate_clip(verts, joints, captions, species, obj_ids, m_lens, 
                                                        obj_transform_trans, obj_transform_rotations, obj_scales, 
                                                        mesh_builder, eval_renderer, smal_model, clip_model)
            clip_score_sum += clip_score.sum().item()
            clip_consistency_sum += clip_consistency.sum().item()
                    
            all_size += motions.shape[0]
        
        obj_pene_ratio_score = obj_pene_ratio_sum / all_size
        obj_pene_ratio_dict[motion_loader_name] = obj_pene_ratio_score

        obj_pene_mean_depth_score = obj_pene_depth_sum / all_size
        obj_pene_mean_depth_dict[motion_loader_name] = obj_pene_mean_depth_score

        obj_pene_max_depth_score = obj_pene_max_sum / all_size
        obj_pene_max_depth_dict[motion_loader_name] = obj_pene_max_depth_score

        clip_score_avg = clip_score_sum / all_size
        clip_score_dict[motion_loader_name] = clip_score_avg

        clip_consistency_avg = clip_consistency_sum / all_size
        clip_consistency_dict[motion_loader_name] = clip_consistency_avg
    
    print(f'---> [{motion_loader_name}] Pene OBJ Ratio: {obj_pene_ratio_score:.4f}')
    print(f'---> [{motion_loader_name}] Pene OBJ Ratio: {obj_pene_ratio_score:.4f}', file=file, flush=True)
    print(f'---> [{motion_loader_name}] Pene OBJ Mean Depth(m): {obj_pene_mean_depth_score:.4f}')
    print(f'---> [{motion_loader_name}] Pene OBJ Mean Depth(m): {obj_pene_mean_depth_score:.4f}', file=file, flush=True)
    print(f'---> [{motion_loader_name}] Pene OBJ Max Depth(m): {obj_pene_max_depth_score:.4f}')
    print(f'---> [{motion_loader_name}] Pene OBJ Max Depth(m): {obj_pene_max_depth_score:.4f}', file=file, flush=True)
    
    print(f'---> [{motion_loader_name}] CLIP Score: {clip_score_avg:.4f}')
    print(f'---> [{motion_loader_name}] CLIP Score: {clip_score_avg:.4f}', file=file, flush=True)
    print(f'---> [{motion_loader_name}] CLIP Consistency: {clip_consistency_avg:.4f}')
    print(f'---> [{motion_loader_name}] CLIP Consistency: {clip_consistency_avg:.4f}', file=file, flush=True)
    
    return obj_pene_ratio_dict, obj_pene_mean_depth_dict, obj_pene_max_depth_dict, clip_score_dict, clip_consistency_dict


def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            _, _, _, _, motions, _, m_lens, *_ = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,
                m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file, diversity_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def evaluate_multimodality(eval_wrapper, mm_motion_loaders, file, mm_num_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                motions, m_lens, *_ = batch # (1, K, T, D)
                motion_embeddings = eval_wrapper.get_motion_embeddings(motions[0], m_lens[0])
                mm_motion_embeddings.append(motion_embeddings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def evaluate_motion_diversity(mm_motion_loaders, std, mean, smal_model, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating Motion Diversity ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_distances = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                motions, m_lens, m_raws, scale_betas = batch # (1, K, T, D)
                motions = motions * std + mean
                motions = motions.float()
                
                motions = motions[0].to(dist_utils.dev())   # (K, N, D)
                scale_betas = scale_betas[0].to(dist_utils.dev())   # (K, 38)
                raw_out = m_raws[0].to(dist_utils.dev())   # (K, 4)
                _, joints, _ = get_verts_from_data(motions, scale_betas, smal_model) # (K, N, J, 3)
                joints = apply_rt_to_motion(joints, raw_out)

                distance = calculate_motion_diversity(joints, m_lens[0])
                mm_distances.append(distance.unsqueeze(0))
        if len(mm_distances) == 0:
            motion_diversity = 0
        else:
            motion_diversity = torch.cat(mm_distances, dim=0).mean().item()
        print(f'---> [{model_name}] Motion Diversity: {motion_diversity:.4f}')
        print(f'---> [{model_name}] Motion Diversity: {motion_diversity:.4f}', file=file, flush=True)
        eval_dict[model_name] = motion_diversity
    return eval_dict


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, diversity_times, mm_num_times, 
               smal_model, mesh_builder, eval_renderer, clip_model, eval_mode, obj_info, run_mm=False):
    
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({'Matching Score': OrderedDict({}), # MM-Dist
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   'MultiModality': OrderedDict({}),
                                   'Skating Ratio': OrderedDict({}),
                                   'Pene Plane Ratio': OrderedDict({}),
                                   'Pene Plane Mean Depth': OrderedDict({}),
                                   'Pene Plane Max Depth': OrderedDict({}),
                                   })
        if eval_mode == 'text_obj':
            all_metrics['Pene OBJ Ratio'] = OrderedDict({})
            all_metrics['Pene OBJ Mean Depth'] = OrderedDict({})
            all_metrics['Pene OBJ Max Depth'] = OrderedDict({})
            all_metrics['CLIP Score'] = OrderedDict({})
            all_metrics['CLIP Consistency'] = OrderedDict({})
            all_metrics['MD'] = OrderedDict({})

        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(eval_wrapper, motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            skating_ratio_dict, pene_plane_ratio_dict, pene_plane_mean_depth_dict, pene_plane_max_depth_dict = evaluate_physic(motion_loaders, smal_model, f)

            if eval_mode == 'text_obj':
                print(f'Time: {datetime.now()}')
                print(f'Time: {datetime.now()}', file=f, flush=True)
                obj_pene_ratio_dict, obj_pene_mean_depth_dict, obj_pene_max_depth_dict, clip_score_dict, clip_consistency_dict = \
                    evaluate_obj_interaction(motion_loaders, smal_model, obj_info, mesh_builder, eval_renderer, clip_model, f)

                print(f'Time: {datetime.now()}')
                print(f'Time: {datetime.now()}', file=f, flush=True)
                md_dict = evaluate_motion_diversity(mm_motion_loaders, gt_loader.dataset.std, gt_loader.dataset.mean, smal_model, f)
            
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f, diversity_times)

            if run_mm:
                print(f'Time: {datetime.now()}')
                print(f'Time: {datetime.now()}', file=f, flush=True)
                mm_score_dict = evaluate_multimodality(eval_wrapper, mm_motion_loaders, f, mm_num_times)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics['Matching Score']:
                    all_metrics['Matching Score'][key] = [item]
                else:
                    all_metrics['Matching Score'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

            for key, item in skating_ratio_dict.items():
                if key not in all_metrics['Skating Ratio']:
                    all_metrics['Skating Ratio'][key] = [item]
                else:
                    all_metrics['Skating Ratio'][key] += [item]

            if eval_mode == 'text_obj':
                
                for key, item in pene_plane_ratio_dict.items():
                    if key not in all_metrics['Pene Plane Ratio']:
                        all_metrics['Pene Plane Ratio'][key] = [item]
                    else:
                        all_metrics['Pene Plane Ratio'][key] += [item]
                
                for key, item in pene_plane_mean_depth_dict.items():
                    if key not in all_metrics['Pene Plane Mean Depth']:
                        all_metrics['Pene Plane Mean Depth'][key] = [item]
                    else:
                        all_metrics['Pene Plane Mean Depth'][key] += [item]
                
                for key, item in pene_plane_max_depth_dict.items():
                    if key not in all_metrics['Pene Plane Max Depth']:
                        all_metrics['Pene Plane Max Depth'][key] = [item]
                    else:
                        all_metrics['Pene Plane Max Depth'][key] += [item]

                for key, item in obj_pene_ratio_dict.items():
                    if key not in all_metrics['Pene OBJ Ratio']:
                        all_metrics['Pene OBJ Ratio'][key] = [item]
                    else:
                        all_metrics['Pene OBJ Ratio'][key] += [item]
                
                for key, item in obj_pene_mean_depth_dict.items():
                    if key not in all_metrics['Pene OBJ Mean Depth']:
                        all_metrics['Pene OBJ Mean Depth'][key] = [item]
                    else:
                        all_metrics['Pene OBJ Mean Depth'][key] += [item]
                
                for key, item in obj_pene_max_depth_dict.items():
                    if key not in all_metrics['Pene OBJ Max Depth']:
                        all_metrics['Pene OBJ Max Depth'][key] = [item]
                    else:
                        all_metrics['Pene OBJ Max Depth'][key] += [item]

                for key, item in clip_score_dict.items():
                    if key not in all_metrics['CLIP Score']:
                        all_metrics['CLIP Score'][key] = [item]
                    else:
                        all_metrics['CLIP Score'][key] += [item]
                
                for key, item in clip_consistency_dict.items(): 
                    if key not in all_metrics['CLIP Consistency']:
                        all_metrics['CLIP Consistency'][key] = [item]
                    else:
                        all_metrics['CLIP Consistency'][key] += [item]

                for key, item in md_dict.items():
                    if key not in all_metrics['MD']:
                        all_metrics['MD'][key] = [item]
                    else:
                        all_metrics['MD'][key] += [item]
            
            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]
            if run_mm:
                for key, item in mm_score_dict.items():
                    if key not in all_metrics['MultiModality']:
                        all_metrics['MultiModality'][key] = [item]
                    else:
                        all_metrics['MultiModality'][key] += [item]


        mean_dict = {}
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)
            for model_name, values in metric_dict.items():
                mean, conf_interval = get_metric_statistics(np.array(values), replication_times)
                mean_dict[metric_name + '_' + model_name] = mean
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)
        return mean_dict


if __name__ == '__main__':
    args = evaluation_parser()
    fixseed(args.seed)
    args.batch_size = 32 # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    log_file = os.path.join(os.path.dirname(args.model_path), 'eval_dogml_obj_{}_{}'.format(name, niter))
    log_file += '.log'
    print(f'Will save to log file [{log_file}]')

    print(f'Eval mode [{args.eval_mode}]')
    num_samples_limit = None  # None means no limit (eval over all dataset)
    run_mm = True
    mm_num_samples = 50
    mm_num_repeats = 10
    mm_num_times = 5
    diversity_times = 256
    replication_times = 5

    dist_utils.setup_dist(args.device)
    logger.configure()

    logger.log("creating data loader...")
    split = 'val'
    gt_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, split=split, mode='gt')
    gen_loader = get_dataset_loader(name=args.dataset, batch_size=16, split=split, mode='eval')

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, gen_loader)

    logger.log(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path, use_avg=args.use_ema)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model, cond_mode=args.eval_mode)   # wrapping model with the classifier-free sampler
    model.to(dist_utils.dev())
    model.eval()

    eval_motion_loaders = {
        'vald': lambda: get_mdm_loader(
            model, diffusion, args,
            gen_loader, mm_num_samples, mm_num_repeats, gen_loader.dataset.opt.max_motion_length, num_samples_limit
        )
    }

    eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_utils.dev())
    
    obj_info = OBJInfo()
    smal_model = SMAL(smal_model_type='39dogs_norm_newv3').to(dist_utils.dev())
    mesh_builder = MeshBatchBuilder(device=dist_utils.dev())
    eval_renderer = SceneRenderer(camera_mode="fix", image_size=(224, 224), device=dist_utils.dev())

    clip_model, _ = clip.load("ViT-B/32", device=dist_utils.dev())
    clip_model.eval()

    evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, diversity_times, 
               mm_num_times, smal_model, mesh_builder, eval_renderer, clip_model, args.eval_mode, obj_info, run_mm)
