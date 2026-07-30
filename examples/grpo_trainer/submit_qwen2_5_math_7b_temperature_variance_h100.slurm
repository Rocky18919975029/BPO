#!/bin/bash
# One-batch pilot/direct temperature-variance experiment on 8xH100.
#
# Submit without --account/--qos on the HKUST-GZ ACD cluster:
#
#   sbatch --partition=acd_u \
#     --export="ALL,MODEL_PATH=/path/to/Qwen2.5-Math-7B,TRAIN_FILE=/path/to/data.parquet,\
#CONDA_ENV_PATH=/path/to/verl" \
#     examples/grpo_trainer/submit_qwen2_5_math_7b_temperature_variance_h100.slurm

#SBATCH --job-name=verl-temp-var
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
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/DAPO-Math-17k.filtered.seed42.sample1536.parquet}
TRAIN_VAL_FILES=${TRAIN_VAL_FILES:-"['${DATA_DIR}/aime24.parquet']"}

if [ ! -f "${TRAIN_FILE}" ]; then
    echo "Training parquet not found: ${TRAIN_FILE}" >&2
    exit 1
fi
if [ ! -f "${DATA_DIR}/aime24.parquet" ]; then
    echo "Validation parquet not found: ${DATA_DIR}/aime24.parquet" >&2
    exit 1
fi
if [ "${HF_HUB_OFFLINE:-1}" = 1 ] && [ ! -d "${MODEL_PATH}" ]; then
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
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-${HF_HUB_OFFLINE}}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
export TRAINER_LOGGER='["console"]'
export WANDB_MODE=disabled

mkdir -p "${RAY_TMP_DIR}"

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
ROLLOUT_N=${ROLLOUT_N:-8}
TEMPERATURES=${TEMPERATURES:-'[0.8,0.9,0.95,1.0,1.05,1.1,1.2]'}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
ROLLOUT_SEED=${ROLLOUT_SEED:-42}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.40}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-1000}
RUN_NAME=${RUN_NAME:-qwen2_5_math_7b_temperature_variance_${SLURM_JOB_ID}}
EXPERIMENT_OUTPUT_DIR=${EXPERIMENT_OUTPUT_DIR:-${VERL_WORKDIR}/temperature_variance_results/${RUN_NAME}}
TEMPERATURE_COUNT=$(
    python - "${TEMPERATURES}" <<'PY'
import ast
import sys

temperatures = ast.literal_eval(sys.argv[1])
if not isinstance(temperatures, list):
    raise SystemExit("TEMPERATURES must be a Python/Hydra list")
print(len(temperatures))
PY
)
CONDITION_COUNT=$((TEMPERATURE_COUNT + 1))

echo "Job ID: ${SLURM_JOB_ID}"
echo "Working directory: ${VERL_WORKDIR}"
echo "Model: ${MODEL_PATH}"
echo "Train file: ${TRAIN_FILE}"
echo "Problems: ${TRAIN_BATCH_SIZE}"
echo "Responses per problem and condition: ${ROLLOUT_N}"
echo "Temperatures: ${TEMPERATURES}"
echo "Conditions: ${CONDITION_COUNT} (one pilot plus ${TEMPERATURE_COUNT} direct)"
echo "Total generated responses: $((TRAIN_BATCH_SIZE * ROLLOUT_N * CONDITION_COUNT))"
echo "Results: ${EXPERIMENT_OUTPUT_DIR}"

MODEL_PATH="${MODEL_PATH}" \
DATA_DIR="${DATA_DIR}" \
TRAIN_FILE="${TRAIN_FILE}" \
TRAIN_VAL_FILES="${TRAIN_VAL_FILES}" \
PROJECT_NAME=temperature_variance_pilot_direct \
RUN_NAME="${RUN_NAME}" \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
ROLLOUT_N="${ROLLOUT_N}" \
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}" \
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}" \
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}" \
ROLLOUT_SEED="${ROLLOUT_SEED}" \
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD}" \
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD}" \
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD}" \
TEMPERATURES="${TEMPERATURES}" \
EXPERIMENT_OUTPUT_DIR="${EXPERIMENT_OUTPUT_DIR}" \
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES}" \
NNODES=1 \
NGPUS_PER_NODE=8 \
bash examples/grpo_trainer/run_qwen2_5_math_7b_temperature_variance_fsdp.sh
