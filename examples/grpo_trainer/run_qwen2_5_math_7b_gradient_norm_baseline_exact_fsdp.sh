#!/usr/bin/env bash
# Exact gradient-norm-weighted GRPO baseline | Qwen2.5-Math-7B | FSDP1
#
# Oracle ablation:
#   1. Compute ||grad_theta log pi_theta(response | prompt)||^2 exactly for
#      every prompt-response pair with a separate FSDP1 backward pass.
#   2. Replace the ordinary group reward mean with
#         b* = sum_i ||g_i||^2 r_i / sum_i ||g_i||^2.
#   3. Keep every other setting from the GRPO baseline unchanged.
#
# The response itself is included in b*. This experiment intentionally does
# not use leave-one-out; it tests whether exact variance reduction improves
# RLVR before introducing a scalable or unbiased estimator.
#
# Exact sensing is expensive. The defaults below form a 64-trajectory pilot
# on 8 GPUs (8 prompts x 8 responses, 8 sensing backwards per rank).

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export ADV_ESTIMATOR=grpo_gradient_norm
export PROJECT_NAME=${PROJECT_NAME:-grpo_gradient_norm_exact_dapo_math17k}
export RUN_NAME=${RUN_NAME:-qwen2_5_math_7b_grpo_gradient_norm_exact_$(date +%Y%m%d_%H%M)}

export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export ROLLOUT_N=${ROLLOUT_N:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-20}
export TEST_FREQ=${TEST_FREQ:-5}

exec bash "${SCRIPT_DIR}/run_qwen2_5_math_7b_reschedule_baseline_fsdp.sh" \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    "$@"
