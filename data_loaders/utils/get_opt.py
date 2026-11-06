from argparse import Namespace
from utils.config import *


def get_opt(dataset_name):
    opt = Namespace()

    opt.dataset_name = dataset_name
    opt.data_root = DATA_ROOT
    opt.motion_dir = MOTION_DIR
    opt.betas_dir = BETA_DIR

    if opt.dataset_name == 'dogml':
        opt.text_dir = TEXT_DIR
        opt.glove_dir = GLOVE_DIR
        opt.max_text_len = 25
    elif opt.dataset_name == 'dogml_obj':
        opt.text_dir = TEXT_INTERACTION_DIR
        opt.glove_dir = GLOVE_INTERACTION_DIR
        opt.obj_dir = OBJ_DIR
        opt.max_text_len = 28
    else:
        raise KeyError('Dataset not recognized')

    opt.joints_num = 35
    opt.dim_pose = 428
    opt.max_motion_length = 300
    opt.min_motion_length = 20
    
    opt.cache_dir = "./cache"
    opt.use_cache = False
    opt.fixed_len = 0
    opt.unit_length = 4

    return opt