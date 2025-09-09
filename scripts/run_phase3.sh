#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

EXP_DIR="experiments/phase3_oneshot_40"
TRAIN_DS="data/shot_train"
TEST_DS="data/shot_test"

BASE_CKPT="experiments/full_training_run/00063/checkpoints/best_model.pt"

# Make sure conda env is active outside or add boilerplate:
# Activate conda env only if not already active
if [ -z "${CONDA_PREFIX:-}" ] || [ "${CONDA_PREFIX##*/}" != "mytimemachine" ]; then
  # Avoid nounset issues from some deactivate hooks by not using -u during activation
  set +u
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate mytimemachine
  set -u
fi

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python scripts/train.py \
  --coach orig_nn \
  --dataset_type ffhq_aging \
  --train_dataset "${TRAIN_DS}" \
  --test_dataset "${TEST_DS}" \
  --exp_dir "${EXP_DIR}" \
  --start_from_encoded_w_plus --train_decoder \
  --batch_size 2 --workers 2 \
  --max_steps 10000 \
  --learning_rate 1.5e-5 \
  --id_lambda 0.3 \
  --lpips_lambda 0.1 --lpips_lambda_aging 0.1 --lpips_lambda_crop 0.8 \
  --l2_lambda 0.1 --l2_lambda_aging 0.25 --l2_lambda_crop 0.5 \
  --w_norm_lambda 0.003 --w_norm_lambda_decoder_scale 0.5 \
  --aging_lambda 5 --aging_lambda_decoder_scale 0.5 \
  --cycle_lambda 1.5 \
  --adaptive_w_norm_lambda 20 \
  --nearest_neighbor_id_loss_lambda 0.05 \
  --contrastive_id_lambda 0.02 \
  --mb_index_path banks/ffhq_ir50_age_5y.pt --mb_use_faiss \
  --mb_top_m 768 --mb_k 48 --mb_min_sim 0.25 --mb_max_sim 0.60 \
  --mb_apply_min_age 38 --mb_apply_max_age 42 --mb_temperature 0.12 \
  --roi_id_lambda_s2 0.05 --roi_use_eyes --roi_use_mouth --roi_use_nose --roi_use_broweyes \
  --roi_size 112 --roi_pad 0.35 --roi_jitter 0.06 \
  --roi_landmarks_model pretrained_models/shape_predictor_68_face_landmarks.dat \
  --extrapolation_start_step 1000000000 \
  --ema --ema_scope decoder --ema_decay 0.999 --eval_with_ema \
  --val_interval 500 --val_deterministic --val_max_batches 2 \
  --resume_checkpoint "${BASE_CKPT}" \
  --target_age_fixed 40 --target_age_jitter 2 \
  | cat


