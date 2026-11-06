# This code is based on https://github.com/openai/guided-diffusion
from utils.fixseed import fixseed
import os
import torch
from utils.parser_utils import generate_args
from utils.model_utils import create_model_and_diffusion, load_saved_model
from utils import dist_utils
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from sample.results_collect import *

from utils.decouple_utils import *
from utils.config import *
from sample.condition import load_condition
from utils.BITE_adapter import SMAL
from utils.BITE_utils import get_verts_from_data
from visualize.render_mesh import render_body_and_object
from utils.decouple_utils import apply_rt_to_motion
import tqdm


def main():
    args = generate_args()
    fixseed(args.seed)
    dist_utils.setup_dist(args.device)
    fps = 30

    print("creating data loader...")
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, split='test', mode='test')

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    print(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path, use_avg=args.use_ema)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model, cond_mode=args.cond_mode)   # wrapping model with the classifier-free sampler
    model.to(dist_utils.dev())
    model.eval()

    output_dir = args.output_dir
    
    for i, (motion, model_kwargs) in enumerate(data):
        out_path = os.path.join(output_dir, f'batch{i}')
        for k, v in model_kwargs['y'].items():
            if torch.is_tensor(v):
                model_kwargs['y'][k] = v.to(dist_utils.dev())

        if args.cond_mode == 'text_obj':
            trans, orient = place_obj_global(motion.shape[0], model_kwargs['y']['obj_center_height'], model_kwargs['y']['obj_scales'])
            model_kwargs['y']['obj_transform_trans'] = trans
            model_kwargs['y']['obj_transform_rotation'] = orient

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['text_guide_scale'] = torch.ones(args.batch_size, device=dist_utils.dev()) * args.guidance_param
        if args.guidance_param_noObj != 1. and 'obj' in args.cond_mode:
            model_kwargs['y']['obj_guide_scale'] = torch.ones(args.batch_size, device=dist_utils.dev()) * args.guidance_param_noObj
        
        # cond_fn = load_condition(args, data.dataset.inner_dataset.inv_transform_th) if args.cond_mode == 'text_obj' else None
        cond_fn = None
        
        smal_model = SMAL(smal_model_type='39dogs_norm_newv3').to(dist_utils.dev())
        
        result_buffers = init_result_buffers()
        
        with torch.no_grad():
            for rep_i in range(args.num_repetitions):
                print(f'Sampling [batch #{i}, repetitions #{rep_i}]')
                
                sample_fn = diffusion.p_sample_loop
                sample, raw_out = sample_fn(
                    model,
                    motion.shape,
                    noise=None,
                    clip_denoised=False,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    skip_timesteps=0,
                )

                # change to dsmal representation
                verts, joints, _ = get_verts_from_data(sample, model_kwargs['y']['scale_betas'], smal_model, data.dataset.inner_dataset.inv_transform_th)
                if raw_out is not None:
                    verts = apply_rt_to_motion(verts, raw_out)
                    joints = apply_rt_to_motion(joints, raw_out)
                                    
                append_rep_results(result_buffers, args.cond_mode, model_kwargs, joints, verts)
                print(f"created {len(result_buffers['motions']) * args.batch_size} samples")

        all_results = finalize_results(result_buffers, args.cond_mode)
        save_results(out_path, args, all_results)

        print(f"saving visualizations to [{out_path}]...")

        for sample_i in range(args.batch_size):
            for rep_i in range(args.num_repetitions):
                
                data_idx, caption, species, verts = get_render_sample(all_results, args.batch_size, sample_i, rep_i)
                obj_id, obj_trans, obj_orient, obj_scale = get_render_obj_info(all_results, args.cond_mode, data_idx)

                mesh_dir = os.path.join(
                    out_path,
                    'mesh_results',
                    'sample{:02d}_rep{:02d}'.format(sample_i, rep_i),
                )

                os.makedirs(mesh_dir, exist_ok=True)
                
                render_body_and_object(verts, smal_model.faces, species, caption, mesh_dir,
                                       obj_id, obj_trans, obj_orient, obj_scale, fps)
                
                os.system('rm -rf %s' % (mesh_dir))

        abs_path = os.path.abspath(out_path)
        print(f'[Done] Results are at [{abs_path}]')

if __name__ == "__main__":
    main()
