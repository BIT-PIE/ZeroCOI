#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${1:-dataset/DogML3D}"
DIGIDOGS_ROOT="${2:-dataset/DigiDogs}"
ANIMO4D_ROOT="${3:-dataset/AniMo4D/data_processed}"
ANIMALML3D_ROOT="${4:-dataset/AnimalML3D/animals_smal_joints}"

echo "[1/7] fitting DigiDogs motions..."
python preprocess/motion_unification/fit.py --dataset DigiDogs --input-dir "$DIGIDOGS_ROOT" --output-dir "$OUTPUT_ROOT"

echo "[2/7] fitting AniMo4D motions..."
python preprocess/motion_unification/fit.py --dataset AniMo4D --input-dir "$ANIMO4D_ROOT" --output-dir "$OUTPUT_ROOT"

echo "[3/7] fitting AnimalML3D motions..."
python preprocess/motion_unification/fit.py --dataset AnimalML3D --input-dir "$ANIMALML3D_ROOT" --output-dir "$OUTPUT_ROOT"

for DATASET in DigiDogs AniMo4D AnimalML3D; do
    case "$DATASET" in
        DigiDogs) STEP=4 ;;
        AniMo4D) STEP=5 ;;
        AnimalML3D) STEP=6 ;;
    esac
    echo "[$STEP/7] unifying and mirroring $DATASET motions..."
    python preprocess/motion_unification/uniform_data.py --dataset "$DATASET" --output-dir "$OUTPUT_ROOT"
done

echo "[7/7] creating train/val/test splits and training statistics..."
python preprocess/motion_unification/cal_mean_variance.py --input-dir "$OUTPUT_ROOT"

echo "Motion Unification Preprocess pipeline completed."
