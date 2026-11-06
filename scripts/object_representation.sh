#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-dataset/DogML3D/obj}"

echo "[1/4] model preprocessing..."
python preprocess/object_representation/model_process.py --root_dir "$ROOT_DIR"

echo "[2/4] bps preprocessing..."
python preprocess/object_representation/bps_process.py --root_dir "$ROOT_DIR"

echo "[3/4] heightmap preprocessing..."
python preprocess/object_representation/heightmap_process.py --root_dir "$ROOT_DIR"

echo "[4/4] sdf preprocessing..."
python preprocess/object_representation/sdf_process.py --root_dir "$ROOT_DIR"

echo "Object Representation Preprocess pipeline completed."
