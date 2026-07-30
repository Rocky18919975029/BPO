#!/bin/bash
# Single-node 8xH100 launcher for problem-balanced positive-only SFT.
#
# Smoke test:
#   sbatch --partition=acd_u --export=ALL,TRAIN_BATCH_SIZE=32,ROLLOUT_N=8,\
#TOTAL_TRAINING_STEPS=1,MODEL_PATH=/path/to/model,TRAIN_FILE=/path/to/train.parquet \
#     examples/grpo_trainer/submit_qwen2_5_math_7b_positive_sft_h100.slurm

#SBATCH --job-name=verl-positive-sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=500G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -xeuo pipefail

export TRAINING_SCRIPT=examples/grpo_trainer/run_qwen2_5_math_7b_positive_sft_fsdp.sh
export PROJECT_NAME=${PROJECT_NAME:-positive_sft_dapo_math17k}
export RUN_NAME=${RUN_NAME:-qwen2_5_math_7b_positive_sft_${SLURM_JOB_ID}}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
export ROLLOUT_N=${ROLLOUT_N:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}

exec bash examples/grpo_trainer/submit_qwen2_5_math_7b_reschedule_baseline_h100.slurm "$@"
