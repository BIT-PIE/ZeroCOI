#!/bin/bash
MODEL_PATH=$1
OUTPUT_DIR=$2
COND_MODE="${3:-text_obj}"

python -m sample.generate \
    --output_dir $OUTPUT_DIR \
    --model_path $MODEL_PATH \
    --cond_mode $COND_MODE \
    --num_repetitions 1 \
    --skip_heavy_init \
    --batch_size 1