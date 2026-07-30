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
"""Numerically stable analysis for the temperature-variance ranking experiment."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def logmeanexp(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    """Compute ``log(mean(exp(values)))`` while preserving all-zero samples as ``-inf``."""
    values = np.asarray(values, dtype=np.float64)
    maximum = np.max(values, axis=axis, keepdims=True)
    finite_maximum = np.isfinite(maximum)
    safe_maximum = np.where(finite_maximum, maximum, 0.0)
    shifted = np.where(finite_maximum, values - safe_maximum, -np.inf)
    mean_exp = np.mean(np.exp(shifted), axis=axis, keepdims=True)
    with np.errstate(divide="ignore"):
        result = np.where(finite_maximum, maximum + np.log(mean_exp), -np.inf)
    return np.squeeze(result, axis=axis)


def estimate_log_j(
    log_is_weights: np.ndarray,
    gradient_update_norm_sq: np.ndarray,
    *,
    weight_power: int,
) -> np.ndarray:
    """Estimate per-problem log J from response samples.

    Args:
        log_is_weights: Shape ``[problems, responses, temperatures]`` or
            ``[problems, responses]``.
        gradient_update_norm_sq: Nonnegative ``||H||^2`` with shape
            ``[problems, responses]``.
        weight_power: One for target-policy pilot samples and two for samples
            drawn directly from the candidate behavior policy.
    """
    if weight_power not in (1, 2):
        raise ValueError(f"weight_power must be 1 or 2, got {weight_power}")

    log_is_weights = np.asarray(log_is_weights, dtype=np.float64)
    gradient_update_norm_sq = np.asarray(gradient_update_norm_sq, dtype=np.float64)
    if log_is_weights.shape[:2] != gradient_update_norm_sq.shape:
        raise ValueError(
            "log_is_weights and gradient_update_norm_sq must share [problems, responses], "
            f"got {log_is_weights.shape} and {gradient_update_norm_sq.shape}"
        )
    if np.any(gradient_update_norm_sq < 0) or np.any(np.isnan(gradient_update_norm_sq)):
        raise ValueError("gradient_update_norm_sq must be nonnegative and cannot contain NaN")

    log_norm_sq = np.full(gradient_update_norm_sq.shape, -np.inf, dtype=np.float64)
    positive = gradient_update_norm_sq > 0
    log_norm_sq[positive] = np.log(gradient_update_norm_sq[positive])
    if log_is_weights.ndim == 3:
        log_norm_sq = log_norm_sq[..., None]
    elif log_is_weights.ndim != 2:
        raise ValueError(f"log_is_weights must have rank 2 or 3, got shape {log_is_weights.shape}")

    log_contribution = weight_power * log_is_weights + log_norm_sq
    return logmeanexp(log_contribution, axis=1)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman_rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Spearman correlation with average ranks for ties."""
    first_ranks = _average_ranks(np.asarray(first, dtype=np.float64))
    second_ranks = _average_ranks(np.asarray(second, dtype=np.float64))
    if np.all(first_ranks == first_ranks[0]) or np.all(second_ranks == second_ranks[0]):
        return math.nan
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def kendall_tau_b(first: np.ndarray, second: np.ndarray) -> float:
    """Kendall's tau-b, including ties, without requiring SciPy."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    concordant = discordant = ties_first = ties_second = 0
    for left in range(len(first)):
        for right in range(left + 1, len(first)):
            first_sign = _comparison_sign(first[left], first[right])
            second_sign = _comparison_sign(second[left], second[right])
            if first_sign == 0 and second_sign == 0:
                continue
            if first_sign == 0:
                ties_first += 1
            elif second_sign == 0:
                ties_second += 1
            elif first_sign == second_sign:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((concordant + discordant + ties_first) * (concordant + discordant + ties_second))
    return (concordant - discordant) / denominator if denominator else math.nan


def pairwise_order_accuracy(first: np.ndarray, second: np.ndarray) -> float:
    """Fraction of temperature pairs whose strict order agrees, ignoring ties."""
    correct = total = 0
    for left in range(len(first)):
        for right in range(left + 1, len(first)):
            first_sign = _comparison_sign(first[left], first[right])
            second_sign = _comparison_sign(second[left], second[right])
            if first_sign == 0 or second_sign == 0:
                continue
            total += 1
            correct += int(first_sign == second_sign)
    return correct / total if total else math.nan


def _comparison_sign(left: float, right: float) -> int:
    if math.isnan(left) or math.isnan(right):
        return 0
    if left == right:
        return 0
    return -1 if left < right else 1


def _finite_mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _safe_exp(value: float) -> float:
    if value == -math.inf:
        return 0.0
    if value >= math.log(np.finfo(np.float64).max):
        return math.inf
    return math.exp(value)


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _bootstrap_relative_log_j(
    per_problem_log_j: np.ndarray,
    reference_index: int,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    problem_count = per_problem_log_j.shape[0]
    bootstrap_values = np.empty((n_bootstrap, per_problem_log_j.shape[1]), dtype=np.float64)
    for bootstrap_index in range(n_bootstrap):
        indices = rng.integers(0, problem_count, size=problem_count)
        curve = logmeanexp(per_problem_log_j[indices], axis=0)
        if np.isfinite(curve[reference_index]):
            bootstrap_values[bootstrap_index] = curve - curve[reference_index]
        else:
            bootstrap_values[bootstrap_index] = np.nan
    return (
        np.nanpercentile(bootstrap_values, 2.5, axis=0),
        np.nanpercentile(bootstrap_values, 97.5, axis=0),
    )


def analyze_temperature_rankings(
    temperatures: list[float],
    pilot_log_j: np.ndarray,
    direct_log_j: np.ndarray,
    *,
    output_dir: str | Path,
    problem_metadata: list[dict[str, Any]] | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare pilot and direct temperature rankings and persist tables/plots."""
    temperatures_array = np.asarray(temperatures, dtype=np.float64)
    pilot_log_j = np.asarray(pilot_log_j, dtype=np.float64)
    direct_log_j = np.asarray(direct_log_j, dtype=np.float64)
    expected_shape = (pilot_log_j.shape[0], len(temperatures))
    if pilot_log_j.shape != expected_shape or direct_log_j.shape != expected_shape:
        raise ValueError(
            f"pilot/direct log J must both have shape [problems, {len(temperatures)}], "
            f"got {pilot_log_j.shape} and {direct_log_j.shape}"
        )
    matching_target = np.flatnonzero(np.isclose(temperatures_array, 1.0, rtol=0, atol=1e-12))
    if len(matching_target) != 1:
        raise ValueError(f"temperatures must contain 1.0 exactly once, got {temperatures}")
    target_index = int(matching_target[0])

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path / "per_problem_log_j.npz",
        temperatures=temperatures_array,
        pilot_log_j=pilot_log_j,
        direct_log_j=direct_log_j,
    )

    batch_pilot = logmeanexp(pilot_log_j, axis=0)
    batch_direct = logmeanexp(direct_log_j, axis=0)
    pilot_relative = (
        batch_pilot - batch_pilot[target_index]
        if np.isfinite(batch_pilot[target_index])
        else np.full_like(batch_pilot, np.nan)
    )
    direct_relative = (
        batch_direct - batch_direct[target_index]
        if np.isfinite(batch_direct[target_index])
        else np.full_like(batch_direct, np.nan)
    )
    pilot_ci_low, pilot_ci_high = _bootstrap_relative_log_j(
        pilot_log_j, target_index, n_bootstrap=n_bootstrap, seed=seed
    )
    direct_ci_low, direct_ci_high = _bootstrap_relative_log_j(
        direct_log_j, target_index, n_bootstrap=n_bootstrap, seed=seed + 1
    )

    curve_rows = []
    for index, temperature in enumerate(temperatures):
        curve_rows.append(
            {
                "temperature": temperature,
                "pilot_log_j": batch_pilot[index],
                "direct_log_j": batch_direct[index],
                "pilot_j_over_tau1": _safe_exp(pilot_relative[index]),
                "direct_j_over_tau1": _safe_exp(direct_relative[index]),
                "pilot_ci95_low": _safe_exp(pilot_ci_low[index]),
                "pilot_ci95_high": _safe_exp(pilot_ci_high[index]),
                "direct_ci95_low": _safe_exp(direct_ci_low[index]),
                "direct_ci95_high": _safe_exp(direct_ci_high[index]),
            }
        )
    with (output_path / "batch_curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    problem_rows = []
    kendall_values: list[float] = []
    spearman_values: list[float] = []
    pairwise_values: list[float] = []
    top1_matches: list[float] = []
    relative_regrets: list[float] = []
    informative_count = 0
    for problem_index in range(pilot_log_j.shape[0]):
        pilot_values = pilot_log_j[problem_index]
        direct_values = direct_log_j[problem_index]
        informative = bool(np.any(np.isfinite(pilot_values)) and np.any(np.isfinite(direct_values)))
        metadata = problem_metadata[problem_index] if problem_metadata is not None else {}
        if informative:
            informative_count += 1
            kendall = kendall_tau_b(pilot_values, direct_values)
            spearman = spearman_rank_correlation(pilot_values, direct_values)
            pairwise = pairwise_order_accuracy(pilot_values, direct_values)
            pilot_best = int(np.argmin(pilot_values))
            direct_best = int(np.argmin(direct_values))
            top1_match = float(pilot_best == direct_best)
            relative_regret = _safe_exp(direct_values[pilot_best] - direct_values[direct_best]) - 1.0
            kendall_values.append(kendall)
            spearman_values.append(spearman)
            pairwise_values.append(pairwise)
            top1_matches.append(top1_match)
            relative_regrets.append(relative_regret)
        else:
            kendall = spearman = pairwise = top1_match = relative_regret = math.nan
            pilot_best = direct_best = -1

        for temperature_index, temperature in enumerate(temperatures):
            problem_rows.append(
                {
                    "problem_index": problem_index,
                    "data_source": metadata.get("data_source"),
                    "temperature": temperature,
                    "pilot_log_j": pilot_values[temperature_index],
                    "direct_log_j": direct_values[temperature_index],
                    "informative": informative,
                    "pilot_best": pilot_best == temperature_index,
                    "direct_best": direct_best == temperature_index,
                    "kendall_tau_b": kendall,
                    "spearman": spearman,
                    "pairwise_accuracy": pairwise,
                    "top1_match": top1_match,
                    "relative_regret": relative_regret,
                }
            )
    with (output_path / "per_problem_rankings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(problem_rows[0]))
        writer.writeheader()
        writer.writerows(problem_rows)

    batch_informative = bool(np.any(np.isfinite(batch_pilot)) and np.any(np.isfinite(batch_direct)))
    batch_pilot_best = int(np.argmin(batch_pilot)) if batch_informative else -1
    batch_direct_best = int(np.argmin(batch_direct)) if batch_informative else -1
    if batch_informative:
        batch_summary = {
            "pilot_best_temperature": temperatures[batch_pilot_best],
            "direct_best_temperature": temperatures[batch_direct_best],
            "top1_match": batch_pilot_best == batch_direct_best,
            "kendall_tau_b": _finite_or_none(kendall_tau_b(batch_pilot, batch_direct)),
            "spearman": _finite_or_none(spearman_rank_correlation(batch_pilot, batch_direct)),
            "pairwise_accuracy": _finite_or_none(pairwise_order_accuracy(batch_pilot, batch_direct)),
            "pilot_selected_direct_j_over_tau1": _finite_or_none(
                _safe_exp(batch_direct[batch_pilot_best] - batch_direct[target_index])
            ),
            "direct_best_j_over_tau1": _finite_or_none(
                _safe_exp(batch_direct[batch_direct_best] - batch_direct[target_index])
            ),
            "pilot_selection_relative_regret": _finite_or_none(
                _safe_exp(batch_direct[batch_pilot_best] - batch_direct[batch_direct_best]) - 1.0
            ),
        }
    else:
        batch_summary = {
            "pilot_best_temperature": None,
            "direct_best_temperature": None,
            "top1_match": None,
            "kendall_tau_b": None,
            "spearman": None,
            "pairwise_accuracy": None,
            "pilot_selected_direct_j_over_tau1": None,
            "direct_best_j_over_tau1": None,
            "pilot_selection_relative_regret": None,
        }
    summary = {
        "temperatures": temperatures,
        "problem_count": int(pilot_log_j.shape[0]),
        "informative_problem_count": informative_count,
        "informative_problem_fraction": informative_count / pilot_log_j.shape[0],
        "batch": batch_summary,
        "per_problem": {
            "kendall_tau_b_mean": _finite_mean(kendall_values),
            "spearman_mean": _finite_mean(spearman_values),
            "pairwise_accuracy_mean": _finite_mean(pairwise_values),
            "top1_match_rate": _finite_mean(top1_matches),
            "relative_regret_mean": _finite_mean(relative_regrets),
        },
    }
    with (output_path / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pilot_ratio = np.exp(np.clip(pilot_relative, -700, 700))
        direct_ratio = np.exp(np.clip(direct_relative, -700, 700))
        figure, axis = plt.subplots(figsize=(8.0, 5.2))
        axis.plot(temperatures, pilot_ratio, marker="o", linewidth=2, label="Pilot estimate")
        axis.plot(temperatures, direct_ratio, marker="s", linewidth=2, label="Direct estimate")
        axis.fill_between(
            temperatures,
            np.exp(np.clip(pilot_ci_low, -700, 700)),
            np.exp(np.clip(pilot_ci_high, -700, 700)),
            alpha=0.16,
        )
        axis.fill_between(
            temperatures,
            np.exp(np.clip(direct_ci_low, -700, 700)),
            np.exp(np.clip(direct_ci_high, -700, 700)),
            alpha=0.16,
        )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.axvline(1.0, color="black", linestyle=":", linewidth=1)
        axis.set_yscale("log")
        axis.set_xlabel("Temperature")
        axis.set_ylabel(r"$J(\tau) / J(1)$")
        axis.set_title("IS-corrected gradient second moment")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_path / "temperature_j_curve.png", dpi=200)
        figure.savefig(output_path / "temperature_j_curve.pdf")
        plt.close(figure)
    except ImportError:
        (output_path / "PLOT_NOT_GENERATED.txt").write_text(
            "matplotlib is not installed. batch_curve.csv contains all curve data.\n"
        )

    return summary
