#!/usr/bin/env bash
# Positive-only SFT | Qwen2.5-Math-7B | DAPO-Math-17k | FSDP
#
# Roll out K responses per problem, discard verifier-negative responses, give
# every active problem equal total weight (1 / number of correct responses),
# and run one full response-balanced SFT epoch over the retained responses.

set -xeuo pipefail

export ADV_ESTIMATOR=positive_sft
export PROJECT_NAME=${PROJECT_NAME:-positive_sft_dapo_math17k}
export RUN_NAME=${RUN_NAME:-qwen2_5_math_7b_positive_sft_$(date +%Y%m%d_%H%M)}

# Recent GRPO/RLOO comparisons use 512 problems x 8 responses.
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
export ROLLOUT_N=${ROLLOUT_N:-8}

# In positive_sft mode this is a count of retained response rows, not prompts.
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}

export POLICY_LOSS_MODE=positive_sft
export ACTOR_LOSS_AGG_MODE=seq-mean-token-mean
export ROLLOUT_LOGPROB_REUSE=True

POSITIVE_REWARD_THRESHOLD=${POSITIVE_REWARD_THRESHOLD:-0.0}

exec bash examples/grpo_trainer/run_qwen2_5_math_7b_reschedule_baseline_fsdp.sh \
    +algorithm.positive_sft_reward_threshold="${POSITIVE_REWARD_THRESHOLD}" \
    "$@"
