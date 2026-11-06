from argparse import ArgumentParser
import argparse
import os
import json


def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    # assert os.path.exists(args_path), 'Arguments json file was not found!'
    if os.path.exists(args_path):
        with open(args_path, 'r') as fr:
            model_args = json.load(fr)
    else:
        model_args = {}

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])
        else:
            pass

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    
    if args.cond_mask_prob_noObj == 0:
        args.guidance_param_noObj = 1
    
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('--model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")
    group.add_argument("--cond_mode", default='text_obj', choices=['text', 'text_obj'], type=str,
                       help="condition mode.")
    group.add_argument("--skip_heavy_init", action='store_true',
                       help="If True, will skip video SDS heavy initialization.")

def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=1000, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")
    group.add_argument("--vsd_cfg_src", default="config_zs.yaml", choices=['config_zs.yaml', 'config_vc2.yaml'], type=str,
                       help="Video SDS config file.")

def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_enc_new_attention',
                       choices=['trans_enc', 'trans_enc_new_attention', 'trans_dec', 'gru', 'hybrid'], type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--emb_trans_dec", default=False, type=bool,
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--layers", default=8, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--cond_mask_prob_noObj", default=.0, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning."
                            "For start and end point as input.")
    group.add_argument("--lambda_mse", default=1.0, type=float, help="MSE loss.")
    group.add_argument("--lambda_approach", default=0.03, type=float, help="Approach loss.")
    group.add_argument("--lambda_pene", default=0.7, type=float, help="SDF penetration loss.")
    group.add_argument("--lambda_contact", default=0.2, type=float, help="Contact loss.")
    group.add_argument("--lambda_vsd", default=0.05, type=float, help="Video SDS loss.")
    group.add_argument("--unconstrained", action='store_true',
                       help="Model is trained unconditionally. That is, it is constrained by neither text nor action. "
                            "Currently tested on HumanAct12 only.")
    group.add_argument("--use_ema", default=True, type=bool,
                       help="If True, will use EMA model averaging.")


def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='dogml', choices=['dogml', 'dogml_obj'], type=str,
                       help="Dataset name (choose from list).")
    group.add_argument("--data_dir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")


def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", required=True, type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", default=True, type=bool,
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--train_platform_type", default='TensorboardPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    group.add_argument("--eval_rep_times", default=3, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=1_000, type=int,
                       help="If -1, will use all samples in the specified split.")
    group.add_argument("--log_interval", default=1_000, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=10_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--num_frames", default=60, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--avg_model_beta", default=0.9999, type=float, help="Average model beta (for EMA).")
    group.add_argument("--adam_beta2", default=0.999, type=float, help="Adam beta2.")


def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=40, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")


def add_edit_options(parser):
    group = parser.add_argument_group('edit')
    group.add_argument("--edit_mode", default='in_between', choices=['in_between', 'upper_body'], type=str,
                       help="Defines which parts of the input motion will be edited.\n"
                            "(1) in_between - suffix and prefix motion taken from input motion, "
                            "middle motion is generated.\n"
                            "(2) upper_body - lower body joints taken from input motion, "
                            "upper body is generated.")
    group.add_argument("--text_condition", default='', type=str,
                       help="Editing will be conditioned on this text prompt. "
                            "If empty, will perform unconditioned editing.")
    group.add_argument("--prefix_end", default=0.25, type=float,
                       help="For in_between editing - Defines the end of input prefix (ratio from all frames).")
    group.add_argument("--suffix_start", default=0.75, type=float,
                       help="For in_between editing - Defines the start of input suffix (ratio from all frames).")


def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--eval_mode", default='text_obj', choices=['text', 'text_obj'], type=str,
                       help="")

def add_guidance_options(parser):
    group = parser.add_argument_group('guidance')
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For text classifier-free sampling - specifies the s parameter.")
    group.add_argument("--guidance_param_noObj", default=1.0, type=float,
                       help="For object classifier-free sampling - specifies the s parameter.")
    group.add_argument("--cond_fn_with_grad", action='store_true', help="use p_sample with gradient to x input.")
    group.add_argument("--guidance_mode", default='', choices=['', 'physical'], type=str, help="guidance type.")
    group.add_argument("--classifiler_scale_app", default=0.0, type=float, help="classifier scale for approach.")
    group.add_argument("--classifiler_scale_pene", default=50.0, type=float, help="classifier scale for sdf pene.")
    group.add_argument("--classifiler_scale_contact", default=50.0, type=float, help="classifier scale for contact.")
    group.add_argument("--classifiler_scale_smooth", default=10.0, type=float, help="classifier scale for smoothness.")
    group.add_argument("--verbose", action='store_true', help="")
    
    group.add_argument("--with_decay", action='store_true', help="")
    group.add_argument("--decay_step", default=0, type=int, help="")
    group.add_argument("--decay_scale", default=1.0, type=float, help="")

def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return parser.parse_args()


def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_guidance_options(parser)
    args = parse_and_load_from_model(parser)

    return args


def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    add_guidance_options(parser)
    return parse_and_load_from_model(parser)