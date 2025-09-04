#!/usr/bin/env bash

set -eo pipefail

BASE_DIR="/home/wczub/RND/AI_DEAGING/repos/mytimemachine"
ENV_NAME="mytimemachine"
EXP_DIR_REL="experiments/full_training_run"
EXP_DIR="$BASE_DIR/$EXP_DIR_REL"
PYTHON_BIN="python"
COACH="orig_nn"

# Shared hparams
WORKERS=2
BATCH_SIZE=2
TEST_BATCH_SIZE=2
TEST_WORKERS=2
VAL_INTERVAL=500
SAVE_INTERVAL=1000
INPUT_NC=4
TARGET_AGE="uniform_random"
CHECKPOINT_PATH="pretrained_models/sam_ffhq_aging.pt"
TRAIN_DATASET="data/train"
SCHEDULER_TYPE="cosine"
GRAD_CLIP_NORM=1.0
WARMUP_STEPS=500
MIN_LR=5e-7

# Stage 1 hparams (Seventh training plan)
ID_LAMBDA_S1=0.35
LPIPS_LAMBDA_S1=0.1
LPIPS_LAMBDA_AGING_S1=0.1
LPIPS_LAMBDA_CROP_S1=0.8
L2_LAMBDA_S1=0.1
L2_LAMBDA_AGING_S1=0.25
L2_LAMBDA_CROP_S1=0.5
W_NORM_LAMBDA_S1=0.003
AGING_LAMBDA_S1=5
# Slightly relax cycle to avoid overemphasis per run 5 plan
CYCLE_LAMBDA_S1=1.5
ADAPTIVE_W_NORM_LAMBDA_S1=20
# Disable extrapolation (interpolation-only) and add NN-ID reg
EXTRAPOLATION_START_STEP_S1=1000000000
EXTRAPOLATION_PROB_START_S1=0.0
EXTRAPOLATION_PROB_END_S1=0.5
NEAREST_NEIGHBOR_ID_LAMBDA_S1=0.2
# Extend Stage 1 duration for identity convergence
MAX_STEPS_S1=35000

# Stage 2 hparams (Seventh training plan)
ID_LAMBDA_S2=0.3
LPIPS_LAMBDA_S2=$LPIPS_LAMBDA_S1
LPIPS_LAMBDA_AGING_S2=$LPIPS_LAMBDA_AGING_S1
LPIPS_LAMBDA_CROP_S2=$LPIPS_LAMBDA_CROP_S1
L2_LAMBDA_S2=$L2_LAMBDA_S1
L2_LAMBDA_AGING_S2=$L2_LAMBDA_AGING_S1
L2_LAMBDA_CROP_S2=$L2_LAMBDA_CROP_S1
W_NORM_LAMBDA_S2=$W_NORM_LAMBDA_S1
W_NORM_LAMBDA_DECODER_SCALE_S2=0.5
AGING_LAMBDA_S2=$AGING_LAMBDA_S1
AGING_LAMBDA_DECODER_SCALE_S2=0.5
CYCLE_LAMBDA_S2=$CYCLE_LAMBDA_S1
ADAPTIVE_W_NORM_LAMBDA_S2=$ADAPTIVE_W_NORM_LAMBDA_S1
# Reduce Stage 2 LR for stability and shorten total steps
LEARNING_RATE_S2=3e-5
MAX_STEPS_S2=45000
NEAREST_NEIGHBOR_ID_LAMBDA_S2=0.05

log() { printf "[two-stage][%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

ensure_conda() {
  # Temporarily disable nounset to avoid failures in system bashrc while initializing conda
  set +u
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
  elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    echo "conda not found in PATH and no conda.sh discovered" >&2
    exit 1
  fi
  conda activate "$ENV_NAME"
  # Re-enable nounset
  set -u
}

run_stage1_phase1() {
  log "Stage 1 Phase 1: 0 → 20000 (baseline S1 settings)"
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  COACH="$COACH" "$PYTHON_BIN" "$BASE_DIR/scripts/train.py" \
    --dataset_type ffhq_aging \
    --workers "$WORKERS" \
    --batch_size "$BATCH_SIZE" \
    --test_batch_size "$TEST_BATCH_SIZE" \
    --test_workers "$TEST_WORKERS" \
    --val_interval "$VAL_INTERVAL" \
    --save_interval "$SAVE_INTERVAL" \
    --start_from_encoded_w_plus \
    --id_lambda "$ID_LAMBDA_S1" \
    --lpips_lambda "$LPIPS_LAMBDA_S1" \
    --lpips_lambda_aging "$LPIPS_LAMBDA_AGING_S1" \
    --lpips_lambda_crop "$LPIPS_LAMBDA_CROP_S1" \
    --l2_lambda "$L2_LAMBDA_S1" \
    --l2_lambda_aging "$L2_LAMBDA_AGING_S1" \
    --l2_lambda_crop "$L2_LAMBDA_CROP_S1" \
    --w_norm_lambda "$W_NORM_LAMBDA_S1" \
    --aging_lambda "$AGING_LAMBDA_S1" \
    --cycle_lambda "$CYCLE_LAMBDA_S1" \
    --input_nc "$INPUT_NC" \
    --target_age "$TARGET_AGE" \
    --use_weighted_id_loss \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --train_dataset "$TRAIN_DATASET" \
    --exp_dir "$EXP_DIR_REL" \
    --adaptive_w_norm_lambda "$ADAPTIVE_W_NORM_LAMBDA_S1" \
    --scheduler_type "$SCHEDULER_TYPE" \
    --warmup_steps "$WARMUP_STEPS" \
    --min_lr "$MIN_LR" \
    --grad_clip_norm "$GRAD_CLIP_NORM" \
    --nan_guard \
    --val_disable_aging \
    --val_deterministic \
    --val_max_batches 2 \
    --val_start_step 2000 \
    --extrapolation_start_step "$EXTRAPOLATION_START_STEP_S1" \
    --extrapolation_prob_start "$EXTRAPOLATION_PROB_START_S1" \
    --extrapolation_prob_end "$EXTRAPOLATION_PROB_END_S1" \
    --nearest_neighbor_id_loss_lambda "$NEAREST_NEIGHBOR_ID_LAMBDA_S1" \
    --train_encoder \
    --max_steps 20000
}

run_stage1_phase2() {
  local resume_ckpt="$1"
  log "Stage 1 Phase 2: 20000 → 30000 (tighten LPIPS crop, lower L2s) from $resume_ckpt"
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  COACH="$COACH" "$PYTHON_BIN" "$BASE_DIR/scripts/train.py" \
    --dataset_type ffhq_aging \
    --workers "$WORKERS" \
    --batch_size "$BATCH_SIZE" \
    --test_batch_size "$TEST_BATCH_SIZE" \
    --test_workers "$TEST_WORKERS" \
    --val_interval "$VAL_INTERVAL" \
    --save_interval "$SAVE_INTERVAL" \
    --start_from_encoded_w_plus \
    --id_lambda "$ID_LAMBDA_S1" \
    --lpips_lambda "$LPIPS_LAMBDA_S1" \
    --lpips_lambda_aging "$LPIPS_LAMBDA_AGING_S1" \
    --lpips_lambda_crop 0.9 \
    --l2_lambda 0.08 \
    --l2_lambda_aging "$L2_LAMBDA_AGING_S1" \
    --l2_lambda_crop 0.4 \
    --w_norm_lambda "$W_NORM_LAMBDA_S1" \
    --aging_lambda "$AGING_LAMBDA_S1" \
    --cycle_lambda "$CYCLE_LAMBDA_S1" \
    --input_nc "$INPUT_NC" \
    --target_age "$TARGET_AGE" \
    --use_weighted_id_loss \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --train_dataset "$TRAIN_DATASET" \
    --exp_dir "$EXP_DIR_REL" \
    --adaptive_w_norm_lambda "$ADAPTIVE_W_NORM_LAMBDA_S1" \
    --scheduler_type "$SCHEDULER_TYPE" \
    --warmup_steps "$WARMUP_STEPS" \
    --min_lr "$MIN_LR" \
    --grad_clip_norm "$GRAD_CLIP_NORM" \
    --nan_guard \
    --val_disable_aging \
    --val_deterministic \
    --val_max_batches 2 \
    --val_start_step 2000 \
    --extrapolation_start_step "$EXTRAPOLATION_START_STEP_S1" \
    --extrapolation_prob_start "$EXTRAPOLATION_PROB_START_S1" \
    --extrapolation_prob_end "$EXTRAPOLATION_PROB_END_S1" \
    --nearest_neighbor_id_loss_lambda "$NEAREST_NEIGHBOR_ID_LAMBDA_S1" \
    --resume_checkpoint "$resume_ckpt" \
    --train_encoder \
    --max_steps 30000
}

run_stage1_phase3() {
  local resume_ckpt="$1"
  log "Stage 1 Phase 3: 30000 → 35000 (NN-ID bump, restore LPIPS/L2) from $resume_ckpt"
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  COACH="$COACH" "$PYTHON_BIN" "$BASE_DIR/scripts/train.py" \
    --dataset_type ffhq_aging \
    --workers "$WORKERS" \
    --batch_size "$BATCH_SIZE" \
    --test_batch_size "$TEST_BATCH_SIZE" \
    --test_workers "$TEST_WORKERS" \
    --val_interval "$VAL_INTERVAL" \
    --save_interval "$SAVE_INTERVAL" \
    --start_from_encoded_w_plus \
    --id_lambda "$ID_LAMBDA_S1" \
    --lpips_lambda "$LPIPS_LAMBDA_S1" \
    --lpips_lambda_aging "$LPIPS_LAMBDA_AGING_S1" \
    --lpips_lambda_crop "$LPIPS_LAMBDA_CROP_S1" \
    --l2_lambda "$L2_LAMBDA_S1" \
    --l2_lambda_aging "$L2_LAMBDA_AGING_S1" \
    --l2_lambda_crop "$L2_LAMBDA_CROP_S1" \
    --w_norm_lambda "$W_NORM_LAMBDA_S1" \
    --aging_lambda "$AGING_LAMBDA_S1" \
    --cycle_lambda "$CYCLE_LAMBDA_S1" \
    --input_nc "$INPUT_NC" \
    --target_age "$TARGET_AGE" \
    --use_weighted_id_loss \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --train_dataset "$TRAIN_DATASET" \
    --exp_dir "$EXP_DIR_REL" \
    --adaptive_w_norm_lambda "$ADAPTIVE_W_NORM_LAMBDA_S1" \
    --scheduler_type "$SCHEDULER_TYPE" \
    --warmup_steps "$WARMUP_STEPS" \
    --min_lr "$MIN_LR" \
    --grad_clip_norm "$GRAD_CLIP_NORM" \
    --nan_guard \
    --val_disable_aging \
    --val_deterministic \
    --val_max_batches 2 \
    --val_start_step 2000 \
    --extrapolation_start_step "$EXTRAPOLATION_START_STEP_S1" \
    --extrapolation_prob_start "$EXTRAPOLATION_PROB_START_S1" \
    --extrapolation_prob_end "$EXTRAPOLATION_PROB_END_S1" \
    --nearest_neighbor_id_loss_lambda 0.25 \
    --resume_checkpoint "$resume_ckpt" \
    --train_encoder \
    --max_steps "$MAX_STEPS_S1"
}

find_latest_ckpt_for_steps() {
  local steps="$1"
  # Example: experiments/full_training_run/000XX/checkpoints/iteration_${steps}.pt
  find "$EXP_DIR" -type f -name "iteration_${steps}.pt" -path '*/checkpoints/*' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | awk '{print $2}'
}

run_stage2() {
  local resume_ckpt="$1"
  log "Starting Stage 2 to ${MAX_STEPS_S2} steps from: $resume_ckpt"
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  COACH="$COACH" "$PYTHON_BIN" "$BASE_DIR/scripts/train.py" \
    --dataset_type ffhq_aging \
    --workers "$WORKERS" \
    --batch_size "$BATCH_SIZE" \
    --test_batch_size "$TEST_BATCH_SIZE" \
    --test_workers "$TEST_WORKERS" \
    --val_interval "$VAL_INTERVAL" \
    --save_interval "$SAVE_INTERVAL" \
    --start_from_encoded_w_plus \
    --id_lambda "$ID_LAMBDA_S2" \
    --lpips_lambda "$LPIPS_LAMBDA_S2" \
    --lpips_lambda_aging "$LPIPS_LAMBDA_AGING_S2" \
    --lpips_lambda_crop "$LPIPS_LAMBDA_CROP_S2" \
    --l2_lambda "$L2_LAMBDA_S2" \
    --l2_lambda_aging "$L2_LAMBDA_AGING_S2" \
    --l2_lambda_crop "$L2_LAMBDA_CROP_S2" \
    --w_norm_lambda "$W_NORM_LAMBDA_S2" \
    --w_norm_lambda_decoder_scale "$W_NORM_LAMBDA_DECODER_SCALE_S2" \
    --aging_lambda "$AGING_LAMBDA_S2" \
    --aging_lambda_decoder_scale "$AGING_LAMBDA_DECODER_SCALE_S2" \
    --cycle_lambda "$CYCLE_LAMBDA_S2" \
    --input_nc "$INPUT_NC" \
    --target_age "$TARGET_AGE" \
    --use_weighted_id_loss \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --train_dataset "$TRAIN_DATASET" \
    --exp_dir "$EXP_DIR_REL" \
    --adaptive_w_norm_lambda "$ADAPTIVE_W_NORM_LAMBDA_S2" \
    --scheduler_type "$SCHEDULER_TYPE" \
    --warmup_steps "$WARMUP_STEPS" \
    --min_lr "$MIN_LR" \
    --grad_clip_norm "$GRAD_CLIP_NORM" \
    --nan_guard \
    --val_disable_aging \
    --val_deterministic \
    --val_max_batches 2 \
    --val_start_step 2000 \
    --extrapolation_start_step "$EXTRAPOLATION_START_STEP_S1" \
    --extrapolation_prob_start "$EXTRAPOLATION_PROB_START_S1" \
    --extrapolation_prob_end "$EXTRAPOLATION_PROB_END_S1" \
    --nearest_neighbor_id_loss_lambda "$NEAREST_NEIGHBOR_ID_LAMBDA_S2" \
    --resume_checkpoint "$resume_ckpt" \
    --train_decoder \
    --max_steps "$MAX_STEPS_S2" \
    --learning_rate "$LEARNING_RATE_S2"
}

main() {
  cd "$BASE_DIR"
  ensure_conda
  run_stage1_phase1

  # Find 20k checkpoint
  ckpt20=$(find_latest_ckpt_for_steps 20000 || true)
  if [[ -z "${ckpt20:-}" || ! -f "$ckpt20" ]]; then
    log "ERROR: Could not locate 20000 checkpoint under $EXP_DIR"
    exit 1
  fi

  run_stage1_phase2 "$ckpt20"

  # Find 30k checkpoint
  ckpt30=$(find_latest_ckpt_for_steps 30000 || true)
  if [[ -z "${ckpt30:-}" || ! -f "$ckpt30" ]]; then
    log "ERROR: Could not locate 30000 checkpoint under $EXP_DIR"
    exit 1
  fi

  run_stage1_phase3 "$ckpt30"

  # Find newest Stage 1 checkpoint for the configured MAX_STEPS_S1
  resume_ckpt=$(find_latest_ckpt_for_steps "$MAX_STEPS_S1" || true)
  if [[ -z "${resume_ckpt:-}" || ! -f "$resume_ckpt" ]]; then
    log "ERROR: Could not locate ${MAX_STEPS_S1} checkpoint under $EXP_DIR"
    exit 1
  fi

  run_stage2 "$resume_ckpt"
}

main "$@"


