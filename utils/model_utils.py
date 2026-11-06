import torch
from model.cmdm import CMDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from utils.config import *

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("unexpected_keys: ", unexpected_keys)
    # assert all([k.startswith('clip_model.') for k in missing_keys])

def load_model_wo_shape(model, state_dict):
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

def load_saved_model(model, model_path, use_avg: bool=False):
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    # Use average model when possible
    if use_avg and 'model_avg' in state_dict.keys():
        print('loading avg model')
        state_dict = state_dict['model_avg']
    else:
        if 'model' in state_dict:
            print('loading model without avg')
            state_dict = state_dict['model']
        else:
            print('checkpoint has no avg model, loading as usual.')
    load_model_wo_clip(model, state_dict)

def create_model_and_diffusion(args, data):
    model = CMDM(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args, data)
    return model, diffusion


def get_model_args(args, data):

    # default args
    clip_version = 'ViT-B/32'
    data_rep = 'hml_vec'
    njoints = 428
    nfeats = 1
    beta_dims = 38

    return {'njoints': njoints, 'nfeats': nfeats,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': args.cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'arch': args.arch,
            'clip_version': clip_version, 'dataset': args.dataset,
            'sample_points_num': SAMPLE_POINTS_NUM, 'beta_dims': beta_dims, 'rt_dim': 4, 
            'cond_mask_prob_noObj':args.cond_mask_prob_noObj, 'inv_transform_th':data.dataset.inner_dataset.inv_transform_th}


def create_gaussian_diffusion(args, data):
    # default params
    predict_xstart = True  # always predict x_start (a.k.a. x0)
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling, for diffusion
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        skip_heavy_init = args.skip_heavy_init,
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        inv_transform_th = data.dataset.inner_dataset.inv_transform_th,
        rescale_timesteps=rescale_timesteps,
        lambda_mse=args.lambda_mse,
        lambda_approach = args.lambda_approach,
        lambda_pene = args.lambda_pene,
        lambda_contact = args.lambda_contact,
        lambda_vsd = args.lambda_vsd,
        vsd_cfg_src = args.vsd_cfg_src,
        cond_mode = args.cond_mode
    )