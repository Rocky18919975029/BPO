import json

import numpy as np
import pytest

from verl.trainer.ppo.temperature_variance import (
    analyze_temperature_rankings,
    estimate_log_j,
    kendall_tau_b,
    logmeanexp,
    pairwise_order_accuracy,
    spearman_rank_correlation,
)
from verl.trainer.ppo.v1.agent_loop_tq import apply_prompt_sampling_overrides


def test_estimate_log_j_matches_pilot_and_direct_definitions():
    gradient_norm_sq = np.array([[1.0, 4.0], [0.0, 9.0]])
    pilot_log_weights = np.log(
        np.array(
            [
                [[1.0, 2.0], [0.5, 1.0]],
                [[4.0, 0.25], [2.0, 3.0]],
            ]
        )
    )

    pilot = np.exp(estimate_log_j(pilot_log_weights, gradient_norm_sq, weight_power=1))
    expected_pilot = np.mean(np.exp(pilot_log_weights) * gradient_norm_sq[..., None], axis=1)
    assert np.allclose(pilot, expected_pilot)

    direct_log_weights = np.log(np.array([[1.0, 0.5], [2.0, 3.0]]))
    direct = np.exp(estimate_log_j(direct_log_weights, gradient_norm_sq, weight_power=2))
    expected_direct = np.mean(np.exp(2 * direct_log_weights) * gradient_norm_sq, axis=1)
    assert np.allclose(direct, expected_direct)


def test_zero_gradient_samples_have_zero_j():
    values = estimate_log_j(
        np.zeros((2, 3, 2)),
        np.zeros((2, 3)),
        weight_power=1,
    )
    assert np.all(np.isneginf(values))
    assert np.isneginf(logmeanexp(np.full((2, 3), -np.inf), axis=1)).all()


def test_rank_metrics_agree_on_identical_and_reverse_orders():
    ascending = np.array([1.0, 2.0, 3.0])
    descending = ascending[::-1]
    assert kendall_tau_b(ascending, ascending) == pytest.approx(1.0)
    assert spearman_rank_correlation(ascending, ascending) == pytest.approx(1.0)
    assert pairwise_order_accuracy(ascending, ascending) == pytest.approx(1.0)
    assert kendall_tau_b(ascending, descending) == pytest.approx(-1.0)
    assert spearman_rank_correlation(ascending, descending) == pytest.approx(-1.0)
    assert pairwise_order_accuracy(ascending, descending) == pytest.approx(0.0)


def test_analysis_writes_curve_and_ranking_outputs(tmp_path):
    temperatures = [0.9, 1.0, 1.1]
    pilot = np.log(np.array([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]]))
    direct = np.log(np.array([[1.5, 3.0, 5.0], [0.5, 2.0, 4.0]]))
    summary = analyze_temperature_rankings(
        temperatures,
        pilot,
        direct,
        output_dir=tmp_path,
        n_bootstrap=20,
        seed=7,
    )

    assert summary["batch"]["pilot_best_temperature"] == 0.9
    assert summary["batch"]["direct_best_temperature"] == 0.9
    assert summary["batch"]["top1_match"]
    assert (tmp_path / "batch_curve.csv").is_file()
    assert (tmp_path / "per_problem_rankings.csv").is_file()
    assert (tmp_path / "per_problem_log_j.npz").is_file()
    saved_summary = json.loads((tmp_path / "summary.json").read_text())
    assert saved_summary["problem_count"] == 2


def test_prompt_temperature_override_is_positive_and_does_not_mutate_defaults():
    prompt = {"__temperature__": 0.9, "uid": "example"}
    defaults = {"temperature": 1.0, "top_p": 1.0}
    overridden = apply_prompt_sampling_overrides(prompt, defaults)
    assert overridden["temperature"] == pytest.approx(0.9)
    assert defaults["temperature"] == pytest.approx(1.0)
    assert "__temperature__" not in prompt

    with pytest.raises(ValueError, match="positive"):
        apply_prompt_sampling_overrides({"__temperature__": 0.0}, defaults)
