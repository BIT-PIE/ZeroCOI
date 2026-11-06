#!/bin/bash
SAVE_DIR=$1
CHECKPOINT=$2

python -m train.train_mdm \
    --dataset dogml_obj \
    --cond_mode 'text_obj' \
    --lr 5e-5 \
    --batch_size 16 \
    --num_steps 50000 \
    --vsd_cfg_src config_zs.yaml \
    --save_dir $SAVE_DIR \
    --resume_checkpoint $CHECKPOINT

