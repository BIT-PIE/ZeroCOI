#!/bin/bash
MODEL_PATH=$1

python -m eval.eval_dogml \
    --model_path $MODEL_PATH \
    --eval_mode 'text_obj' \
    --skip_heavy_init