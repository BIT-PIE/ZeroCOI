#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${1:-dataset/DogML3D}"

echo "[1/2] normalizing captions and building base GloVe..."
python preprocess/semantic_augmentation/normalize.py --output-root "$OUTPUT_ROOT"

echo "[2/2] augmenting captions and building interaction GloVe..."
python preprocess/semantic_augmentation/augment.py --output-root "$OUTPUT_ROOT"

echo "Semantic Augmentation Preprocess pipeline completed."
