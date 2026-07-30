#!/usr/bin/env bash
# GRPO baseline | Qwen2.5-Math-7B | DAPO-Math-17k | FSDP | NVIDIA GPUs
#
# Reproduce the pure GRPO baseline used for comparison with Re-Schedule-style
# experiments. This script intentionally enables no dynamic weighting, HPF,
# mixed-policy, or transition-aware behavior.
#
# Modes:
#   MODE=train  Train for TOTAL_TRAINING_STEPS rollout/update steps.
#   MODE=eval   Validate on the six paper evaluation datasets.

set -xeuo pipefail

MODE=${MODE:-train}
ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}

PROJECT_NAME=${PROJECT_NAME:-grpo_dapo_math17k_reschedule_baseline}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-Math-7B}
DATA_DIR=${DATA_DIR:-./datasets}
RUN_NAME=${RUN_NAME:-qwen2_5_math_7b_grpo_baseline_$(date +%Y%m%d_%H%M)}

TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/DAPO-Math-17k.parquet}
TRAIN_VAL_FILES=${TRAIN_VAL_FILES:-"['${DATA_DIR}/aime24.parquet']"}
PAPER_EVAL_FILES=${PAPER_EVAL_FILES:-"['${DATA_DIR}/aime24.parquet','${DATA_DIR}/aime25.parquet','${DATA_DIR}/amc23.parquet','${DATA_DIR}/math500.parquet','${DATA_DIR}/minerva_math.parquet','${DATA_DIR}/olympiadbench.parquet']"}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Comparable baseline: 512 problems x 32 trajectories = 16,384 rows/step.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
ROLLOUT_N=${ROLLOUT_N:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-50}

ACTOR_LR=${ACTOR_LR:-1e-6}
ACTOR_LOSS_AGG_MODE=${ACTOR_LOSS_AGG_MODE:-token-mean}
POLICY_LOSS_MODE=${POLICY_LOSS_MODE:-vanilla}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
ROLLOUT_TP=${ROLLOUT_TP:-4}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}
ROLLOUT_TRAIN_TEMPERATURE=${ROLLOUT_TRAIN_TEMPERATURE:-1.0}
ROLLOUT_TRAIN_TOP_P=${ROLLOUT_TRAIN_TOP_P:-1.0}
ROLLOUT_VAL_TEMPERATURE=${ROLLOUT_VAL_TEMPERATURE:-1.0}
ROLLOUT_VAL_TOP_P=${ROLLOUT_VAL_TOP_P:-0.7}
ROLLOUT_VAL_N=${ROLLOUT_VAL_N:-1}
ROLLOUT_SEED=${ROLLOUT_SEED:-42}
ROLLOUT_LOGPROB_REUSE=${ROLLOUT_LOGPROB_REUSE:-True}
REWARD_STRICT_BOX_VERIFY=${REWARD_STRICT_BOX_VERIFY:-True}

ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-False}

TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb"]'}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-300}
export WANDB_TIMEOUT=${WANDB_TIMEOUT:-300}
export WANDB_RETRY_DELAY=${WANDB_RETRY_DELAY:-60}
export WANDB_MAX_RETRIES=${WANDB_MAX_RETRIES:-10}

export PYTHONHASHSEED=${PYTHONHASHSEED:-42}
export PYTORCH_SEED=${PYTORCH_SEED:-42}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

# Include optimizer and dataloader state so RESUME_MODE=auto is a real resume.
SAVE_CONTENTS=${SAVE_CONTENTS:-"['model','optimizer','extra','hf_model']"}
SAVE_FREQ=${SAVE_FREQ:-1}
RESUME_MODE=${RESUME_MODE:-disable}

COMMON_DATA=(
    algorithm.adv_estimator=${ADV_ESTIMATOR}
    algorithm.use_kl_in_reward=False
    data.train_files="['${TRAIN_FILE}']"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=left
    data.shuffle=True
    data.seed=42
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    +actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPLEMENTATION}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.shuffle=False
    actor_rollout_ref.actor.loss_agg_mode=${ACTOR_LOSS_AGG_MODE}
    actor_rollout_ref.actor.policy_loss.loss_mode=${POLICY_LOSS_MODE}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}
    actor_rollout_ref.actor.checkpoint.save_contents="${SAVE_CONTENTS}"
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TRAIN_TEMPERATURE}
    actor_rollout_ref.rollout.top_p=${ROLLOUT_TRAIN_TOP_P}
    actor_rollout_ref.rollout.seed=${ROLLOUT_SEED}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.val_kwargs.n=${ROLLOUT_VAL_N}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=${ROLLOUT_VAL_TEMPERATURE}
    actor_rollout_ref.rollout.val_kwargs.top_p=${ROLLOUT_VAL_TOP_P}
)

# Reuse vLLM behavior log-probs as old_log_probs while retaining PPO clipping.
ROLLOUT_CORRECTION=(
    algorithm.rollout_correction.bypass_mode=${ROLLOUT_LOGPROB_REUSE}
    algorithm.rollout_correction.loss_type=ppo_clip
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD}
)

REWARD=(
    reward_model.reward_manager=dapo
    reward.reward_manager.name=dapo
    +reward_model.reward_kwargs.strict_box_verify=${REWARD_STRICT_BOX_VERIFY}
    +reward.reward_kwargs.strict_box_verify=${REWARD_STRICT_BOX_VERIFY}
)

RAY=(
    +ray_kwargs.ray_init.address=local
    +ray_kwargs.ray_init._temp_dir="${RAY_TMP_DIR:-/tmp/ray_${USER:-unknown}_${RUN_NAME}}"
)

TRAINER_COMMON=(
    trainer.balance_batch=True
    trainer.logger="${TRAINER_LOGGER}"
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${RUN_NAME}"
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.critic_warmup=0
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ:-1}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.resume_mode=${RESUME_MODE}
)

case "${MODE}" in
    train)
        DATA=("${COMMON_DATA[@]}" data.val_files="${TRAIN_VAL_FILES}")
        TRAINER=(
            "${TRAINER_COMMON[@]}"
            trainer.val_before_train=False
            trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
            trainer.rollout_data_dir="${ROLLOUT_DATA_DIR:-./rollout_data/${PROJECT_NAME}/${RUN_NAME}}"
            trainer.validation_data_dir="${VALIDATION_DATA_DIR:-./validation_data/${PROJECT_NAME}/${RUN_NAME}}"
        )
        ;;
    eval)
        DATA=("${COMMON_DATA[@]}" data.val_files="${VAL_FILES:-${PAPER_EVAL_FILES}}")
        TRAINER=(
            "${TRAINER_COMMON[@]}"
            trainer.val_before_train=True
            trainer.val_only=True
            trainer.rollout_data_dir="${ROLLOUT_DATA_DIR:-./rollout_data/${PROJECT_NAME}/${RUN_NAME}_eval}"
            trainer.validation_data_dir="${VALIDATION_DATA_DIR:-./validation_data/${PROJECT_NAME}/${RUN_NAME}_eval}"
        )
        ;;
    *)
        echo "Unknown MODE=${MODE}; expected train or eval" >&2
        exit 1
        ;;
esac

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${ROLLOUT_CORRECTION[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${RAY[@]}" \
    "${TRAINER[@]}" \
    "$@"
