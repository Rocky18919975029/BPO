# Copyright 2026 The BPO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Frozen-snapshot pilot/direct temperature-variance ranking experiment."""

from __future__ import annotations

import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transfer_queue as tq
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.temperature_variance import analyze_temperature_rankings, estimate_log_j
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils import tensordict_utils as tu
from verl.utils.skip import SkipManager
from verl.workers.utils.padding import response_from_nested


class PPOTrainerTemperatureVariance(PPOTrainerSync):
    """Run one frozen problem batch without taking an optimizer step."""

    def fit(self, agent_loop_manager) -> None:
        self.agent_loop_manager = agent_loop_manager
        SkipManager.init(self.config)
        SkipManager.set_step(self.global_steps)

        experiment = self.config.temperature_variance_experiment
        temperatures = [float(value) for value in experiment.temperatures]
        response_count = int(experiment.responses_per_problem)
        output_dir = Path(str(experiment.output_dir)).expanduser().resolve()
        bootstrap_samples = int(experiment.get("bootstrap_samples", 1000))
        bootstrap_seed = int(experiment.get("bootstrap_seed", 42))
        self._validate_experiment_settings(temperatures, response_count)
        output_dir.mkdir(parents=True, exist_ok=True)

        if int(self.config.actor_rollout_ref.rollout.n) != response_count:
            raise ValueError(
                "actor_rollout_ref.rollout.n must equal "
                f"temperature_variance_experiment.responses_per_problem ({response_count})"
            )
        if not math.isclose(float(self.config.actor_rollout_ref.rollout.temperature), 1.0):
            raise ValueError("The base rollout temperature must be 1.0 for target-policy pilot sampling")
        if int(self.config.actor_rollout_ref.rollout.top_k) != -1:
            raise ValueError("The temperature-only experiment requires actor_rollout_ref.rollout.top_k=-1")
        if not math.isclose(float(self.config.actor_rollout_ref.rollout.top_p), 1.0):
            raise ValueError("The temperature-only experiment requires actor_rollout_ref.rollout.top_p=1.0")

        problem_count = int(self.config.data.train_batch_size)
        print(
            "[temperature-variance] starting frozen-snapshot experiment: "
            f"{problem_count} problems x {response_count} responses, temperatures={temperatures}",
            flush=True,
        )

        base_prompts = self._next_train_batch(problem_count)
        condition_batches: list[tuple[str, float, KVBatchMeta, dict[str, int]]] = []
        conditions = [("pilot", 1.0)] + [("direct", temperature) for temperature in temperatures]

        # Keep vLLM awake while all conditions are generated. Actor/FSDP sensing
        # starts only after every response is resident in the CPU TransferQueue.
        for condition_index, (phase, temperature) in enumerate(conditions, start=1):
            condition_batch, uid_to_problem = self._make_condition_batch(
                base_prompts,
                temperature=temperature,
                response_count=response_count,
            )
            self._submit_batch_to_rollout(condition_batch)
            generated_batch, _ = self.replay_buffer.sample(
                global_steps=self.global_steps,
                partition_id="train",
                batch_size=problem_count,
            )
            expected_responses = problem_count * response_count
            if len(generated_batch) != expected_responses:
                raise RuntimeError(
                    f"{phase} tau={temperature} produced {len(generated_batch)} trajectories; "
                    f"expected {expected_responses}"
                )
            condition_batches.append((phase, temperature, generated_batch, uid_to_problem))
            print(
                f"[temperature-variance] rollout condition {condition_index}/{len(conditions)} complete: "
                f"phase={phase}, tau={temperature}, responses={len(generated_batch)}",
                flush=True,
            )

        self.on_sample_end()

        pilot_log_weights = np.empty((problem_count, response_count, len(temperatures)), dtype=np.float64)
        pilot_update_norm_sq = np.empty((problem_count, response_count), dtype=np.float64)
        direct_log_j = np.empty((problem_count, len(temperatures)), dtype=np.float64)
        problem_metadata: list[dict[str, Any]] = [{} for _ in range(problem_count)]
        samples_path = output_dir / "samples.jsonl"

        with samples_path.open("w") as samples_file:
            for condition_index, (phase, temperature, batch, uid_to_problem) in enumerate(condition_batches, start=1):
                print(
                    f"[temperature-variance] sensing condition {condition_index}/{len(condition_batches)}: "
                    f"phase={phase}, tau={temperature}",
                    flush=True,
                )
                if self.reward_loop_manager.reward_loop_worker_handles is None:
                    batch = self._compute_reward_colocate(batch)
                batch = self._balance_batch(batch, metrics={}, logging_prefix=f"{phase}_tau_{temperature}")

                if phase == "pilot":
                    response_log_probs = {
                        candidate: self._compute_response_log_probs(batch, candidate) for candidate in temperatures
                    }
                else:
                    response_log_probs = {1.0: self._compute_response_log_probs(batch, 1.0)}
                    if temperature != 1.0:
                        response_log_probs[temperature] = self._compute_response_log_probs(batch, temperature)

                batch = self._compute_exact_gradient_norm(batch, metrics={})
                measured = self._extract_condition_measurements(
                    batch,
                    uid_to_problem=uid_to_problem,
                    response_count=response_count,
                    response_log_probs=response_log_probs,
                )

                if phase == "pilot":
                    target_log_prob = measured["response_log_probs"][1.0]
                    pilot_update_norm_sq[:] = measured["gradient_update_norm_sq"]
                    for temperature_index, candidate in enumerate(temperatures):
                        pilot_log_weights[..., temperature_index] = (
                            target_log_prob - measured["response_log_probs"][candidate]
                        )
                    self._write_pilot_samples(
                        samples_file,
                        measured,
                        temperatures=temperatures,
                        pilot_log_weights=pilot_log_weights,
                        problem_metadata=problem_metadata,
                    )
                else:
                    target_log_prob = measured["response_log_probs"][1.0]
                    behavior_log_prob = measured["response_log_probs"][temperature]
                    log_weights = target_log_prob - behavior_log_prob
                    temperature_index = temperatures.index(temperature)
                    direct_log_j[:, temperature_index] = estimate_log_j(
                        log_weights,
                        measured["gradient_update_norm_sq"],
                        weight_power=2,
                    )
                    self._write_direct_samples(
                        samples_file,
                        measured,
                        temperature=temperature,
                        log_weights=log_weights,
                        problem_metadata=problem_metadata,
                    )

                tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)
                print(
                    f"[temperature-variance] sensing condition {condition_index}/{len(condition_batches)} complete",
                    flush=True,
                )

        pilot_log_j = estimate_log_j(
            pilot_log_weights,
            pilot_update_norm_sq,
            weight_power=1,
        )
        summary = analyze_temperature_rankings(
            temperatures,
            pilot_log_j,
            direct_log_j,
            output_dir=output_dir,
            problem_metadata=problem_metadata,
            n_bootstrap=bootstrap_samples,
            seed=bootstrap_seed,
        )
        with (output_dir / "resolved_experiment_config.json").open("w") as handle:
            json.dump(
                {
                    "temperatures": temperatures,
                    "problem_count": problem_count,
                    "responses_per_problem": response_count,
                    "model_path": str(self.config.actor_rollout_ref.model.path),
                    "train_files": list(self.config.data.train_files),
                    "rollout_seed": int(self.config.actor_rollout_ref.rollout.seed),
                    "max_prompt_length": int(self.config.data.max_prompt_length),
                    "max_response_length": int(self.config.data.max_response_length),
                },
                handle,
                indent=2,
            )
        self._shutdown_dump_executor()
        print("[temperature-variance] experiment complete", flush=True)
        print(json.dumps(summary, indent=2), flush=True)
        print(f"[temperature-variance] results: {output_dir}", flush=True)

    @staticmethod
    def _validate_experiment_settings(temperatures: list[float], response_count: int) -> None:
        if response_count <= 1:
            raise ValueError(f"responses_per_problem must be greater than one, got {response_count}")
        if not temperatures:
            raise ValueError("temperature_variance_experiment.temperatures cannot be empty")
        if any(not math.isfinite(value) or value <= 0 for value in temperatures):
            raise ValueError(f"All temperatures must be finite and positive, got {temperatures}")
        if len(set(temperatures)) != len(temperatures):
            raise ValueError(f"Temperatures must be unique, got {temperatures}")
        if sum(math.isclose(value, 1.0, rel_tol=0, abs_tol=1e-12) for value in temperatures) != 1:
            raise ValueError(f"Temperatures must contain target temperature 1.0 exactly once, got {temperatures}")

    def _make_condition_batch(
        self,
        base_prompts,
        *,
        temperature: float,
        response_count: int,
    ):
        batch = base_prompts.clone()
        uids = [uuid.uuid4().hex for _ in range(len(batch))]
        tu.assign_non_tensor_stack(batch, "uid", uids)
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        tu.assign_non_tensor_data(batch, "__rollout_n__", response_count)
        tu.assign_non_tensor_data(batch, "__temperature__", temperature)
        return batch, {uid: problem_index for problem_index, uid in enumerate(uids)}

    def _compute_response_log_probs(self, batch: KVBatchMeta, temperature: float) -> np.ndarray:
        batch.extra_info.update(
            {
                "calculate_entropy": False,
                "compute_loss": False,
                "temperature": float(temperature),
            }
        )
        output: KVBatchMeta = self.actor_rollout_wg.compute_log_prob(batch)
        if len(output) != len(batch):
            raise RuntimeError(f"Log-probability worker returned {len(output)} rows for a batch of {len(batch)}")
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["log_probs", "response_mask"],
        )
        response_log_probs = response_from_nested(data["log_probs"], data["response_mask"])
        return np.asarray([float(values.sum()) for values in response_log_probs.unbind()], dtype=np.float64)

    def _extract_condition_measurements(
        self,
        batch: KVBatchMeta,
        *,
        uid_to_problem: dict[str, int],
        response_count: int,
        response_log_probs: dict[float, np.ndarray],
    ) -> dict[str, Any]:
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=[
                "uid",
                "prompts",
                "responses",
                "response_mask",
                "rm_scores",
                "rollout_log_probs",
                "score_grad_norm_sq",
                "data_source",
            ],
        )
        rewards = np.asarray([float(values.sum()) for values in data["rm_scores"].unbind()], dtype=np.float64)
        rollout_log_probs = np.asarray(
            [float(values.sum()) for values in data["rollout_log_probs"].unbind()],
            dtype=np.float64,
        )
        score_norm_sq = data["score_grad_norm_sq"].reshape(-1).to(torch.float64).cpu().numpy()
        prompts = list(data["prompts"].unbind())
        responses = list(data["responses"].unbind())
        # TransferQueue may return non-tensor columns as LinkedList,
        # NonTensorStack, or numpy arrays depending on the backend.
        data_sources = list(data["data_source"])

        shape = (len(uid_to_problem), response_count)
        result: dict[str, Any] = {
            "reward": np.empty(shape, dtype=np.float64),
            "score_grad_norm_sq": np.empty(shape, dtype=np.float64),
            "gradient_update_norm_sq": np.empty(shape, dtype=np.float64),
            "rollout_response_log_prob": np.empty(shape, dtype=np.float64),
            "prompt_text": np.empty(shape, dtype=object),
            "response_text": np.empty(shape, dtype=object),
            "response_length": np.empty(shape, dtype=np.int64),
            "data_source": np.empty(shape, dtype=object),
            "response_log_probs": {
                temperature: np.empty(shape, dtype=np.float64) for temperature in response_log_probs
            },
        }
        seen = np.zeros(shape, dtype=bool)
        for row_index, (key, tag) in enumerate(zip(batch.keys, batch.tags, strict=True)):
            if tag.get("is_padding", False):
                continue
            key_parts = key.rsplit("_", 2)
            if len(key_parts) != 3:
                raise RuntimeError(f"Unexpected trajectory key format: {key}")
            uid, session_id, output_index = key_parts
            if int(output_index) != 0:
                raise RuntimeError("The experiment currently supports the single-turn agent loop only")
            problem_index = uid_to_problem[uid]
            response_index = int(session_id)
            if not 0 <= response_index < response_count:
                raise RuntimeError(f"Unexpected response index {response_index} in key {key}")
            if seen[problem_index, response_index]:
                raise RuntimeError(f"Duplicate response slot for key {key}")
            seen[problem_index, response_index] = True

            reward = rewards[row_index]
            norm_sq = score_norm_sq[row_index]
            result["reward"][problem_index, response_index] = reward
            result["score_grad_norm_sq"][problem_index, response_index] = norm_sq
            result["gradient_update_norm_sq"][problem_index, response_index] = reward * reward * norm_sq
            result["rollout_response_log_prob"][problem_index, response_index] = rollout_log_probs[row_index]
            result["prompt_text"][problem_index, response_index] = self.tokenizer.decode(
                prompts[row_index], skip_special_tokens=True
            )
            result["response_text"][problem_index, response_index] = self.tokenizer.decode(
                responses[row_index], skip_special_tokens=True
            )
            result["response_length"][problem_index, response_index] = int(responses[row_index].numel())
            result["data_source"][problem_index, response_index] = data_sources[row_index]
            for temperature, values in response_log_probs.items():
                result["response_log_probs"][temperature][problem_index, response_index] = values[row_index]

        if not np.all(seen):
            missing = np.argwhere(~seen)
            raise RuntimeError(f"Missing {len(missing)} response slots; first missing slot={missing[0].tolist()}")
        return result

    @staticmethod
    def _write_json_line(handle, payload: dict[str, Any]) -> None:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + os.linesep)

    def _write_pilot_samples(
        self,
        handle,
        measured: dict[str, Any],
        *,
        temperatures: list[float],
        pilot_log_weights: np.ndarray,
        problem_metadata: list[dict[str, Any]],
    ) -> None:
        for problem_index in range(pilot_log_weights.shape[0]):
            problem_metadata[problem_index] = {
                "data_source": str(measured["data_source"][problem_index, 0]),
                "prompt": measured["prompt_text"][problem_index, 0],
            }
            for response_index in range(pilot_log_weights.shape[1]):
                self._write_json_line(
                    handle,
                    {
                        "phase": "pilot",
                        "problem_index": problem_index,
                        "response_index": response_index,
                        "sampling_temperature": 1.0,
                        "data_source": str(measured["data_source"][problem_index, response_index]),
                        "prompt": measured["prompt_text"][problem_index, response_index],
                        "response": measured["response_text"][problem_index, response_index],
                        "response_length": int(measured["response_length"][problem_index, response_index]),
                        "reward": float(measured["reward"][problem_index, response_index]),
                        "score_grad_norm_sq": float(measured["score_grad_norm_sq"][problem_index, response_index]),
                        "gradient_update_norm_sq": float(
                            measured["gradient_update_norm_sq"][problem_index, response_index]
                        ),
                        "target_response_log_prob": float(
                            measured["response_log_probs"][1.0][problem_index, response_index]
                        ),
                        "rollout_response_log_prob": float(
                            measured["rollout_response_log_prob"][problem_index, response_index]
                        ),
                        "rollout_minus_recomputed_sampling_log_prob": float(
                            measured["rollout_response_log_prob"][problem_index, response_index]
                            - measured["response_log_probs"][1.0][problem_index, response_index]
                        ),
                        "candidate_response_log_probs": {
                            str(temperature): float(
                                measured["response_log_probs"][temperature][problem_index, response_index]
                            )
                            for temperature in temperatures
                        },
                        "candidate_log_is_weights": {
                            str(temperature): float(pilot_log_weights[problem_index, response_index, temperature_index])
                            for temperature_index, temperature in enumerate(temperatures)
                        },
                    },
                )

    def _write_direct_samples(
        self,
        handle,
        measured: dict[str, Any],
        *,
        temperature: float,
        log_weights: np.ndarray,
        problem_metadata: list[dict[str, Any]],
    ) -> None:
        for problem_index in range(log_weights.shape[0]):
            if not problem_metadata[problem_index]:
                problem_metadata[problem_index] = {
                    "data_source": str(measured["data_source"][problem_index, 0]),
                    "prompt": measured["prompt_text"][problem_index, 0],
                }
            for response_index in range(log_weights.shape[1]):
                self._write_json_line(
                    handle,
                    {
                        "phase": "direct",
                        "problem_index": problem_index,
                        "response_index": response_index,
                        "sampling_temperature": temperature,
                        "data_source": str(measured["data_source"][problem_index, response_index]),
                        "prompt": measured["prompt_text"][problem_index, response_index],
                        "response": measured["response_text"][problem_index, response_index],
                        "response_length": int(measured["response_length"][problem_index, response_index]),
                        "reward": float(measured["reward"][problem_index, response_index]),
                        "score_grad_norm_sq": float(measured["score_grad_norm_sq"][problem_index, response_index]),
                        "gradient_update_norm_sq": float(
                            measured["gradient_update_norm_sq"][problem_index, response_index]
                        ),
                        "target_response_log_prob": float(
                            measured["response_log_probs"][1.0][problem_index, response_index]
                        ),
                        "behavior_response_log_prob": float(
                            measured["response_log_probs"][temperature][problem_index, response_index]
                        ),
                        "rollout_response_log_prob": float(
                            measured["rollout_response_log_prob"][problem_index, response_index]
                        ),
                        "rollout_minus_recomputed_sampling_log_prob": float(
                            measured["rollout_response_log_prob"][problem_index, response_index]
                            - measured["response_log_probs"][temperature][problem_index, response_index]
                        ),
                        "log_is_weight": float(log_weights[problem_index, response_index]),
                    },
                )
