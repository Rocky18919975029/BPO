#!/usr/bin/env bash
# Frozen-snapshot temperature ranking experiment | Qwen2.5-Math-7B | FSDP1
#
# For one fixed batch of problems:
#   1. sample N target-policy (temperature=1) pilot responses;
#   2. estimate J(tau) for every candidate from those same pilot responses;
#   3. independently sample N responses from every candidate temperature;
#   4. estimate J(tau) directly and compare the two rankings.
#
# No optimizer step is taken and no checkpoint is written.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export ADV_ESTIMATOR=grpo
export PROJECT_NAME=${PROJECT_NAME:-temperature_variance_pilot_direct}
export RUN_NAME=${RUN_NAME:-qwen2_5_math_7b_temperature_variance_$(date +%Y%m%d_%H%M)}
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console"]'}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
export ROLLOUT_N=${ROLLOUT_N:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
export TOTAL_TRAINING_STEPS=1
export TEST_FREQ=-1
export SAVE_FREQ=-1
export RESUME_MODE=disable
export ROLLOUT_TRAIN_TEMPERATURE=1.0
export ROLLOUT_TRAIN_TOP_P=1.0
export ROLLOUT_LOGPROB_REUSE=False

TEMPERATURES=${TEMPERATURES:-'[0.8,0.9,0.95,1.0,1.05,1.1,1.2]'}
EXPERIMENT_OUTPUT_DIR=${EXPERIMENT_OUTPUT_DIR:-"./temperature_variance_results/${RUN_NAME}"}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-1000}

exec bash "${SCRIPT_DIR}/run_qwen2_5_math_7b_reschedule_baseline_fsdp.sh" \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.top_k=-1 \
    trainer.rollout_data_dir=null \
    trainer.validation_data_dir=null \
    +temperature_variance_experiment.enabled=True \
    +temperature_variance_experiment.temperatures="${TEMPERATURES}" \
    +temperature_variance_experiment.responses_per_problem="${ROLLOUT_N}" \
    +temperature_variance_experiment.output_dir="${EXPERIMENT_OUTPUT_DIR}" \
    +temperature_variance_experiment.bootstrap_samples="${BOOTSTRAP_SAMPLES}" \
    +temperature_variance_experiment.bootstrap_seed="${ROLLOUT_SEED:-42}" \
    "$@"
