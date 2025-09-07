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

# Stage 1 hparams (Eleventh training plan: EMA + ROI schedule)
ID_LAMBDA_S1=0.3
LPIPS_LAMBDA_S1=0.1
LPIPS_LAMBDA_AGING_S1=0.1
LPIPS_LAMBDA_CROP_S1=0.8
L2_LAMBDA_S1=0.1
L2_LAMBDA_AGING_S1=0.25
L2_LAMBDA_CROP_S1=0.5
W_NORM_LAMBDA_S1=0.003
AGING_LAMBDA_S1=5
CYCLE_LAMBDA_S1=1.5
ADAPTIVE_W_NORM_LAMBDA_S1=20
# Disable extrapolation (interpolation-only) and add NN-ID reg
EXTRAPOLATION_START_STEP_S1=1000000000
EXTRAPOLATION_PROB_START_S1=0.0
EXTRAPOLATION_PROB_END_S1=0.5
NEAREST_NEIGHBOR_ID_LAMBDA_S1=0.1
# Stage 1 duration (extended)
MAX_STEPS_S1=40000

# Contrastive impostor loss (FAISS miner)
CONTRASTIVE_ID_LAMBDA_S1=0.04
CONTRASTIVE_ID_LAMBDA_S2=0.02
MB_INDEX_PATH="banks/ffhq_ir50_age_5y.pt"
# FAISS miner presets (Twelfth plan: softening v2)
MB_K=48
MB_APPLY_MIN_AGE=35
MB_APPLY_MAX_AGE=45
MB_BIN_NEIGHBOR_RADIUS=0
MB_TEMPERATURE=0.12
MB_USE_FAISS=1
MB_TOP_M=768
MB_MIN_SIM=0.25
MB_MAX_SIM=0.60

# ROI-ID micro loss (eyes + mouth)
ROI_ID_LAMBDA=0.05
ROI_USE_EYES=1
ROI_USE_MOUTH=1
ROI_SIZE=112
ROI_PAD=0.35
ROI_JITTER=0.06
ROI_LANDMARKS_MODEL="pretrained_models/shape_predictor_68_face_landmarks.dat"

# EMA controls (enable EMA on decoder, use during eval)
EMA_ENABLE=1
EMA_DECAY=0.999
EMA_SCOPE="decoder"
EVAL_WITH_EMA=1

# ROI schedule controls (S1 schedule; S2 fixed)
ROI_S1_SCHEDULE="0:0.05,20000:0.07,36000:0.05"
ROI_ID_LAMBDA_S2=0.05

# Stage 2 hparams (Ninth training plan)
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
# Stage 2 LR and duration (shortened total)
LEARNING_RATE_S2=3e-5
MAX_STEPS_S2=55000
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
  log "Stage 1: 0 → ${MAX_STEPS_S1} (Eleventh: FAISS + ROI-ID + EMA + ROI schedule)"
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
    --contrastive_id_lambda "$CONTRASTIVE_ID_LAMBDA_S1" \
    --mb_index_path "$MB_INDEX_PATH" \
    --mb_k "$MB_K" \
    --mb_use_faiss \
    --mb_top_m "$MB_TOP_M" \
    --mb_min_sim "$MB_MIN_SIM" \
    --mb_max_sim "$MB_MAX_SIM" \
    --mb_apply_min_age "$MB_APPLY_MIN_AGE" \
    --mb_apply_max_age "$MB_APPLY_MAX_AGE" \
    --mb_bin_neighbor_radius "$MB_BIN_NEIGHBOR_RADIUS" \
    --mb_temperature "$MB_TEMPERATURE" \
    --roi_id_lambda "$ROI_ID_LAMBDA" \
    $( [[ "$ROI_USE_EYES" == "1" ]] && echo "--roi_use_eyes" ) \
    $( [[ "$ROI_USE_MOUTH" == "1" ]] && echo "--roi_use_mouth" ) \
    --roi_size "$ROI_SIZE" \
    --roi_pad "$ROI_PAD" \
    --roi_jitter "$ROI_JITTER" \
    --roi_landmarks_model "$ROI_LANDMARKS_MODEL" \
    --roi_id_schedule_s1 "$ROI_S1_SCHEDULE" \
    --geom_lambda 0.3 --geom_stage s1 --geom_parts eyes,nose,mouth --geom_weights 1.0,0.6,0.4 --geom_huber_delta 0.03 \
    --age_anchor_path anchors/actor_w_age5.pt \
    --age_anchor_lambda 0.02 --age_anchor_stage s1 --age_anchor_space w --age_anchor_bin_size 5 \
    --seed 123 \
    $( [[ "$EMA_ENABLE" == "1" ]] && echo "--ema" ) \
    --ema_scope "$EMA_SCOPE" \
    --ema_decay "$EMA_DECAY" \
    $( [[ "$EVAL_WITH_EMA" == "1" ]] && echo "--eval_with_ema" ) \
    --train_encoder \
    --max_steps "$MAX_STEPS_S1"
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
    --roi_id_schedule_s1 "$ROI_S1_SCHEDULE" \
    $( [[ "$EMA_ENABLE" == "1" ]] && echo "--ema" ) \
    --ema_scope "$EMA_SCOPE" \
    --ema_decay "$EMA_DECAY" \
    $( [[ "$EVAL_WITH_EMA" == "1" ]] && echo "--eval_with_ema" ) \
    --seed 123 \
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
    --roi_id_schedule_s1 "$ROI_S1_SCHEDULE" \
    $( [[ "$EMA_ENABLE" == "1" ]] && echo "--ema" ) \
    --ema_scope "$EMA_SCOPE" \
    --ema_decay "$EMA_DECAY" \
    $( [[ "$EVAL_WITH_EMA" == "1" ]] && echo "--eval_with_ema" ) \
    --seed 123 \
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
    --contrastive_id_lambda "$CONTRASTIVE_ID_LAMBDA_S2" \
    --mb_index_path "$MB_INDEX_PATH" \
    --mb_k "$MB_K" \
    --mb_use_faiss \
    --mb_top_m "$MB_TOP_M" \
    --mb_min_sim "$MB_MIN_SIM" \
    --mb_max_sim "$MB_MAX_SIM" \
    --mb_apply_min_age "$MB_APPLY_MIN_AGE" \
    --mb_apply_max_age "$MB_APPLY_MAX_AGE" \
    --mb_bin_neighbor_radius "$MB_BIN_NEIGHBOR_RADIUS" \
    --mb_temperature "$MB_TEMPERATURE" \
    --roi_id_lambda_s2 "$ROI_ID_LAMBDA_S2" \
    $( [[ "$ROI_USE_EYES" == "1" ]] && echo "--roi_use_eyes" ) \
    $( [[ "$ROI_USE_MOUTH" == "1" ]] && echo "--roi_use_mouth" ) \
    --roi_size "$ROI_SIZE" \
    --roi_pad "$ROI_PAD" \
    --roi_jitter "$ROI_JITTER" \
    --roi_landmarks_model "$ROI_LANDMARKS_MODEL" \
    $( [[ "$EMA_ENABLE" == "1" ]] && echo "--ema" ) \
    --ema_scope "$EMA_SCOPE" \
    --ema_decay "$EMA_DECAY" \
    $( [[ "$EVAL_WITH_EMA" == "1" ]] && echo "--eval_with_ema" ) \
    --resume_checkpoint "$resume_ckpt" \
    --train_decoder \
    --max_steps "$MAX_STEPS_S2" \
    --learning_rate "$LEARNING_RATE_S2"
}

main() {
  cd "$BASE_DIR"
  ensure_conda
  run_stage1_phase1

  # Find newest Stage 1 checkpoint for the configured MAX_STEPS_S1
  resume_ckpt=$(find_latest_ckpt_for_steps "$MAX_STEPS_S1" || true)
  if [[ -z "${resume_ckpt:-}" || ! -f "$resume_ckpt" ]]; then
    log "ERROR: Could not locate ${MAX_STEPS_S1} checkpoint under $EXP_DIR"
    exit 1
  fi

  run_stage2 "$resume_ckpt"
}

main "$@"


