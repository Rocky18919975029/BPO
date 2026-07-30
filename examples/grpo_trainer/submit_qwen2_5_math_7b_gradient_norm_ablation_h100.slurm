#!/bin/bash
# Matched 8xH100 comparison with four available baseline modes:
#   grpo: ordinary full-group reward mean
#   grpo_loo: strict ordinary leave-one-out reward mean (no reward std)
#   gradient_norm: full-group exact gradient-norm-weighted reward mean
#   gradient_norm_loo: strict exact gradient-norm-weighted LOO mean (no reward std)
#
# Submit two jobs with identical model/data paths and seeds:
#
#   sbatch --partition=<partition> --account=<account> --export=ALL,\
#ABLATION_VARIANT=grpo,MODEL_PATH=/path/to/Qwen2.5-Math-7B,DATA_DIR=/path/to/data \
#     examples/grpo_trainer/submit_qwen2_5_math_7b_gradient_norm_ablation_h100.slurm
#
#   sbatch --partition=<partition> --account=<account> --export=ALL,\
#ABLATION_VARIANT=gradient_norm,MODEL_PATH=/path/to/Qwen2.5-Math-7B,DATA_DIR=/path/to/data \
#     examples/grpo_trainer/submit_qwen2_5_math_7b_gradient_norm_ablation_h100.slurm
#
# Start with TOTAL_TRAINING_STEPS=1 for a cluster smoke test. The local
# development machine is not expected to have a GPU.
#
# For rapid code-path checks, set FAST_SMOKE=1. This uses 4 prompts x 2
# responses, caps responses at 128 tokens, runs one exact-gradient round per
# rank, and disables validation, checkpoints, W&B, and trajectory dumps.

#SBATCH --job-name=verl-grpo-gn-ablate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=500G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -xeuo pipefail

VERL_WORKDIR=${VERL_WORKDIR:-${SLURM_SUBMIT_DIR}}
ABLATION_VARIANT=${ABLATION_VARIANT:-gradient_norm}
FAST_SMOKE=${FAST_SMOKE:-0}

case "${FAST_SMOKE,,}" in
    0|false|no)
        FAST_SMOKE=0
        ;;
    1|true|yes)
        FAST_SMOKE=1
        ;;
    *)
        echo "Invalid FAST_SMOKE=${FAST_SMOKE}; expected 0/1, false/true, or no/yes" >&2
        exit 1
        ;;
esac

case "${ABLATION_VARIANT}" in
    grpo)
        TRAINING_SCRIPT=examples/grpo_trainer/run_qwen2_5_math_7b_reschedule_baseline_fsdp.sh
        ADV_ESTIMATOR=grpo
        DEFAULT_PROJECT_NAME=grpo_exact_norm_ablation
        DEFAULT_RUN_PREFIX=qwen2_5_math_7b_grpo_control
        ADVANTAGE_ARGS=()
        ;;
    grpo_loo)
        TRAINING_SCRIPT=examples/grpo_trainer/run_qwen2_5_math_7b_reschedule_baseline_fsdp.sh
        ADV_ESTIMATOR=grpo_loo
        DEFAULT_PROJECT_NAME=grpo_exact_norm_loo_ablation
        DEFAULT_RUN_PREFIX=qwen2_5_math_7b_grpo_strict_loo_control
        ADVANTAGE_ARGS=(algorithm.norm_adv_by_std_in_grpo=False)
        ;;
    gradient_norm)
        TRAINING_SCRIPT=examples/grpo_trainer/run_qwen2_5_math_7b_gradient_norm_baseline_exact_fsdp.sh
        ADV_ESTIMATOR=grpo_gradient_norm
        DEFAULT_PROJECT_NAME=grpo_exact_norm_ablation
        DEFAULT_RUN_PREFIX=qwen2_5_math_7b_grpo_gradient_norm_exact
        ADVANTAGE_ARGS=()
        ;;
    gradient_norm_loo)
        TRAINING_SCRIPT=examples/grpo_trainer/run_qwen2_5_math_7b_gradient_norm_baseline_exact_fsdp.sh
        ADV_ESTIMATOR=grpo_gradient_norm_loo
        DEFAULT_PROJECT_NAME=grpo_exact_norm_loo_ablation
        DEFAULT_RUN_PREFIX=qwen2_5_math_7b_grpo_gradient_norm_strict_loo_exact
        ADVANTAGE_ARGS=(algorithm.norm_adv_by_std_in_grpo=False)
        ;;
    *)
        echo "Unknown ABLATION_VARIANT=${ABLATION_VARIANT}; expected grpo, grpo_loo, gradient_norm, or gradient_norm_loo" >&2
        exit 1
        ;;
esac

# Optional cluster module setup.
if command -v module >/dev/null 2>&1; then
    module purge
    if [ -n "${GCC_MODULE:-}" ]; then
        module load "${GCC_MODULE}"
    fi
    if [ -n "${CUDA_MODULE:-}" ]; then
        module load "${CUDA_MODULE}"
    fi
    module list
fi

# Optionally activate a conda environment. Otherwise use the submitted
# shell's current Python environment.
if [ -n "${CONDA_ENV_PATH:-}" ]; then
    CONDA_SH=${CONDA_SH:-$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh}
    if [ ! -f "${CONDA_SH}" ]; then
        echo "Conda init script not found: ${CONDA_SH}" >&2
        exit 1
    fi
    # shellcheck source=/dev/null
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV_PATH}"
    hash -r
fi

cd "${VERL_WORKDIR}"

python - <<'PY'
import sys

import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("No CUDA device visible inside the Slurm allocation.")
if torch.cuda.device_count() != 8:
    raise SystemExit(f"Expected 8 GPUs, found {torch.cuda.device_count()}.")
print("gpu:", torch.cuda.get_device_name(0))
PY

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-Math-7B}
DATA_DIR=${DATA_DIR:-${VERL_WORKDIR}/datasets}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/DAPO-Math-17k.parquet}
TRAIN_VAL_FILES=${TRAIN_VAL_FILES:-"['${DATA_DIR}/aime24.parquet']"}

if [ ! -f "${TRAIN_FILE}" ]; then
    echo "Training parquet not found: ${TRAIN_FILE}" >&2
    exit 1
fi
if [ ! -f "${DATA_DIR}/aime24.parquet" ]; then
    echo "Validation parquet not found: ${DATA_DIR}/aime24.parquet" >&2
    exit 1
fi
if [ "${HF_HUB_OFFLINE:-0}" = 1 ] && [ ! -d "${MODEL_PATH}" ]; then
    echo "HF_HUB_OFFLINE=1 requires a local MODEL_PATH directory: ${MODEL_PATH}" >&2
    exit 1
fi

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
export RAY_TMP_DIR=${RAY_TMP_DIR:-${VERL_WORKDIR}/ray_tmp/${SLURM_JOB_ID}}
if [ "${FAST_SMOKE}" = 1 ]; then
    export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console"]'}
    export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
else
    export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb"]'}
    export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
fi
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-${VERL_WORKDIR}/wandb}
export WANDB_SILENT=${WANDB_SILENT:-true}
export WANDB_START_METHOD=${WANDB_START_METHOD:-thread}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-${HF_HUB_OFFLINE}}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}

mkdir -p "${WANDB_DIR}" "${RAY_TMP_DIR}"

PROJECT_NAME=${PROJECT_NAME:-${DEFAULT_PROJECT_NAME}}
RUN_NAME=${RUN_NAME:-${DEFAULT_RUN_PREFIX}_${SLURM_JOB_ID}}

if [ "${FAST_SMOKE}" = 1 ]; then
    # Eight trajectories keep all eight actor ranks active while preserving a
    # nontrivial two-response GRPO group.
    TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
    ROLLOUT_N=${ROLLOUT_N:-2}
    PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
    PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
    LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
    MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
    MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-128}
    TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
    TEST_FREQ=${TEST_FREQ:--1}
    SAVE_FREQ=${SAVE_FREQ:--1}
    EXTRA_TRAINING_ARGS=(
        trainer.rollout_data_dir=null
        trainer.validation_data_dir=null
    )
else
    # Matched pilot defaults. Both variants see 8 prompts x 8 responses per step.
    TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
    ROLLOUT_N=${ROLLOUT_N:-8}
    PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
    PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
    LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
    MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
    MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
    TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-20}
    TEST_FREQ=${TEST_FREQ:-5}
    SAVE_FREQ=${SAVE_FREQ:-1}
    EXTRA_TRAINING_ARGS=()
fi

# Exact no_sync gradients temporarily materialize a full gradient on every
# rank. Offloading optimizer state and using conservative vLLM placement leave
# headroom for this oracle pass on 80GB H100s.
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.40}

SAVE_CONTENTS=${SAVE_CONTENTS:-"['model','optimizer','extra','hf_model']"}
RESUME_MODE=${RESUME_MODE:-disable}

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Working directory: ${VERL_WORKDIR}"
echo "Ablation variant: ${ABLATION_VARIANT}"
echo "Advantage estimator: ${ADV_ESTIMATOR}"
echo "Strict LOO: $([[ ${ABLATION_VARIANT} == *_loo ]] && echo true || echo false)"
echo "Training script: ${TRAINING_SCRIPT}"
echo "Fast smoke mode: ${FAST_SMOKE}"
echo "Project: ${PROJECT_NAME}"
echo "Run: ${RUN_NAME}"
echo "Model: ${MODEL_PATH}"
echo "Data dir: ${DATA_DIR}"
echo "Train batch size: ${TRAIN_BATCH_SIZE}"
echo "Rollout N: ${ROLLOUT_N}"
echo "Total trajectories per step: $((TRAIN_BATCH_SIZE * ROLLOUT_N))"
echo "Max prompt length: ${MAX_PROMPT_LENGTH}"
echo "Max response length: ${MAX_RESPONSE_LENGTH}"
echo "Training steps: ${TOTAL_TRAINING_STEPS}"
echo "Validation frequency: ${TEST_FREQ}"
echo "Checkpoint frequency: ${SAVE_FREQ}"
echo "Rollout seed: ${ROLLOUT_SEED:-42}"
echo "Checkpoint contents: ${SAVE_CONTENTS}"
echo "Resume mode: ${RESUME_MODE}"

MODEL_PATH="${MODEL_PATH}" \
DATA_DIR="${DATA_DIR}" \
TRAIN_FILE="${TRAIN_FILE}" \
TRAIN_VAL_FILES="${TRAIN_VAL_FILES}" \
ADV_ESTIMATOR="${ADV_ESTIMATOR}" \
PROJECT_NAME="${PROJECT_NAME}" \
RUN_NAME="${RUN_NAME}" \
MODE=train \
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
TEST_FREQ="${TEST_FREQ}" \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
ROLLOUT_N="${ROLLOUT_N}" \
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE}" \
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}" \
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}" \
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD}" \
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD}" \
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD}" \
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
ROLLOUT_LOGPROB_REUSE="${ROLLOUT_LOGPROB_REUSE:-True}" \
REWARD_STRICT_BOX_VERIFY="${REWARD_STRICT_BOX_VERIFY:-True}" \
ROLLOUT_SEED="${ROLLOUT_SEED:-42}" \
SAVE_CONTENTS="${SAVE_CONTENTS}" \
SAVE_FREQ="${SAVE_FREQ}" \
RESUME_MODE="${RESUME_MODE}" \
NNODES=1 \
NGPUS_PER_NODE=8 \
bash "${TRAINING_SCRIPT}" "${EXTRA_TRAINING_ARGS[@]}" "${ADVANTAGE_ARGS[@]}" "$@"
