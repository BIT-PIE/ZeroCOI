from os.path import join as pjoin

DATA_ROOT = "./dataset/DogML3D"

MOTION_DIR = pjoin(DATA_ROOT, 'new_joint_vecs')
BETA_DIR = pjoin(DATA_ROOT, 'scale_betas')
OBJ_DIR = pjoin(DATA_ROOT, 'obj')
DSMAL_TEX_DIR = pjoin(DATA_ROOT, 'dog_textures')

TEXT_DIR = pjoin(DATA_ROOT, 'texts')
GLOVE_DIR = pjoin(DATA_ROOT, 'glove')
TEXT_INTERACTION_DIR = pjoin(DATA_ROOT, 'texts_interaction')
GLOVE_INTERACTION_DIR = pjoin(DATA_ROOT, 'interaction_glove')

SAMPLE_POINTS_NUM=1024

text_mot_path = './text_mot_match/finest_aug.tar'

dsmal_kinematic_chain = [
        [0, 25, 26, 27, 28, 29, 30, 31],
        [0, 21, 22, 23, 24],
        [0, 17, 18, 19, 20],
        [0, 1, 2, 3, 4, 5, 6, 15, 16, 32],
        [6, 11, 12, 13, 14],
        [6, 7, 8, 9, 10],
        [16, 33],
        [16, 34]
    ]