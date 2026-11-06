#!/bin/bash
SAVE_DIR=$1
CHECKPOINT=$2

python -m train.train_mdm \
    --dataset dogml \
    --cond_mode 'text' \
    --lr 1e-4 \
    --batch_size 64 \
    --num_steps 800000 \
    --save_dir $SAVE_DIR \
    --resume_checkpoint $CHECKPOINT

