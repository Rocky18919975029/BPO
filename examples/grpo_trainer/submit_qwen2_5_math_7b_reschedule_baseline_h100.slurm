#!/bin/bash
# Single-node 8xH100 launcher for the Qwen2.5-Math-7B GRPO baseline.
#
# Configure cluster-specific values at submission time, for example:
#   sbatch --partition=<partition> --account=<account> --export=ALL,\
#MODEL_PATH=/path/to/Qwen2.5-Math-7B,DATA_DIR=/path/to/datasets \
#     examples/grpo_trainer/submit_qwen2_5_math_7b_reschedule_baseline_h100.slurm
#
# One-step smoke test:
#   sbatch --partition=<partition> --export=ALL,TRAIN_BATCH_SIZE=32,\
#TOTAL_TRAINING_STEPS=1,ROLLOUT_N=32,MODEL_PATH=/path/to/model,DATA_DIR=/path/to/data \
#     examples/grpo_trainer/submit_qwen2_5_math_7b_reschedule_baseline_h100.slurm

#SBATCH --job-name=verl-resched-grpo
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

# Optional cluster module setup. Leave unset when the active environment already
# provides a compatible compiler and CUDA runtime.
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

# Optionally activate a conda environment. Otherwise use the submitted shell's
# current Python environment.
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
print("gpu:", torch.cuda.get_device_name(0))
PY

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-Math-7B}
DATA_DIR=${DATA_DIR:-${VERL_WORKDIR}/datasets}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/DAPO-Math-17k.parquet}
MODE=${MODE:-train}

if [ ! -f "${TRAIN_FILE}" ]; then
    echo "Training parquet not found: ${TRAIN_FILE}" >&2
    exit 1
fi

required_val_files=("${DATA_DIR}/aime24.parquet")
if [ "${MODE}" = eval ]; then
    required_val_files+=(
        "${DATA_DIR}/aime25.parquet"
        "${DATA_DIR}/amc23.parquet"
        "${DATA_DIR}/math500.parquet"
        "${DATA_DIR}/minerva_math.parquet"
        "${DATA_DIR}/olympiadbench.parquet"
    )
fi
for val_file in "${required_val_files[@]}"; do
    if [ ! -f "${val_file}" ]; then
        echo "Validation parquet not found: ${val_file}" >&2
        exit 1
    fi
done

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
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb"]'}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-${VERL_WORKDIR}/wandb}
export WANDB_SILENT=${WANDB_SILENT:-true}
export WANDB_START_METHOD=${WANDB_START_METHOD:-thread}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-${HF_HUB_OFFLINE}}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}

mkdir -p "${WANDB_DIR}" "${RAY_TMP_DIR}"

PROJECT_NAME=${PROJECT_NAME:-grpo_dapo_math17k_reschedule_baseline}
RUN_NAME=${RUN_NAME:-qwen2_5_math_7b_grpo_baseline_${SLURM_JOB_ID}}

# Safe colocated defaults for a single 8xH100 node. Override these to reproduce
# a different engineering placement without changing the GRPO algorithm.
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}

ROLLOUT_N=${ROLLOUT_N:-32}
SAVE_CONTENTS=${SAVE_CONTENTS:-"['model','optimizer','extra','hf_model']"}
SAVE_FREQ=${SAVE_FREQ:-1}
RESUME_MODE=${RESUME_MODE:-disable}

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Working directory: ${VERL_WORKDIR}"
echo "Project: ${PROJECT_NAME}"
echo "Run: ${RUN_NAME}"
echo "Model: ${MODEL_PATH}"
echo "Data dir: ${DATA_DIR}"
echo "Mode: ${MODE}"
echo "Training steps: ${TOTAL_TRAINING_STEPS:-150}"
echo "Rollout N: ${ROLLOUT_N}"
echo "Rollout seed: ${ROLLOUT_SEED:-42}"
echo "Reuse rollout log-probs: ${ROLLOUT_LOGPROB_REUSE:-True}"
echo "Strict boxed verifier: ${REWARD_STRICT_BOX_VERIFY:-True}"
echo "Checkpoint contents: ${SAVE_CONTENTS}"
echo "Resume mode: ${RESUME_MODE}"

MODEL_PATH="${MODEL_PATH}" \
DATA_DIR="${DATA_DIR}" \
TRAIN_FILE="${TRAIN_FILE}" \
PROJECT_NAME="${PROJECT_NAME}" \
RUN_NAME="${RUN_NAME}" \
MODE="${MODE}" \
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-150}" \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}" \
ROLLOUT_N="${ROLLOUT_N}" \
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}" \
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}" \
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD}" \
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD}" \
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD}" \
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
ROLLOUT_LOGPROB_REUSE="${ROLLOUT_LOGPROB_REUSE:-True}" \
REWARD_STRICT_BOX_VERIFY="${REWARD_STRICT_BOX_VERIFY:-True}" \
SAVE_CONTENTS="${SAVE_CONTENTS}" \
SAVE_FREQ="${SAVE_FREQ}" \
RESUME_MODE="${RESUME_MODE}" \
NNODES=1 \
NGPUS_PER_NODE=8 \
bash examples/grpo_trainer/run_qwen2_5_math_7b_reschedule_baseline_fsdp.sh "$@"
