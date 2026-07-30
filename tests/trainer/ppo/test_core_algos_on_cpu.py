# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import random
import unittest

import numpy as np
import pytest
import torch

import verl.trainer.ppo.core_algos
from verl.trainer.ppo.core_algos import (
    compute_gae_advantage_return,
    compute_grpo_gradient_norm_loo_outcome_advantage,
    compute_grpo_gradient_norm_outcome_advantage,
    compute_grpo_loo_outcome_advantage,
    compute_grpo_outcome_advantage,
    compute_grpo_vectorized_outcome_advantage,
    compute_rloo_outcome_advantage,
    compute_rloo_vectorized_outcome_advantage,
    get_adv_estimator_fn,
    kl_penalty,
    register_adv_est,
)


def mock_test_fn():
    pass


class TestRegisterAdvEst(unittest.TestCase):
    def setUp(self):
        """Clear the registry before each test"""
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY = {
            "gae": lambda x: x * 2,
            "vtrace": lambda x: x + 1,
        }
        self.ADV_ESTIMATOR_REGISTRY = verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY

    def tearDown(self) -> None:
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        return super().tearDown()

    def test_register_new_function(self):
        """Test registering a new function with a string name"""

        @register_adv_est("test_estimator")
        def test_fn():
            pass

        self.assertIn("test_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_estimator"], test_fn)

    def test_register_with_enum(self):
        """Test registering with an enum value (assuming AdvantageEstimator exists)"""
        from enum import Enum

        class AdvantageEstimator(Enum):
            TEST = "test_enum_estimator"

        @register_adv_est(AdvantageEstimator.TEST)
        def test_fn():
            pass

        self.assertIn("test_enum_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_enum_estimator"], test_fn)

    def test_duplicate_registration_same_function(self):
        """Test that registering the same function twice doesn't raise an error"""
        register_adv_est("duplicate_test")(mock_test_fn)
        register_adv_est("duplicate_test")(mock_test_fn)

        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["duplicate_test"], mock_test_fn)

    def test_duplicate_registration_different_function(self):
        """Test that registering different functions with same name raises ValueError"""

        @register_adv_est("conflict_test")
        def test_fn1():
            pass

        with self.assertRaises(ValueError):

            @register_adv_est("conflict_test")
            def test_fn2():
                pass

    def test_decorator_preserves_function(self):
        """Test that the decorator returns the original function"""

        def test_fn():
            return "original"

        decorated = register_adv_est("preserve_test")(test_fn)
        self.assertEqual(decorated(), "original")

    def test_multiple_registrations(self):
        """Test registering multiple different functions"""
        init_adv_count = len(self.ADV_ESTIMATOR_REGISTRY)

        @register_adv_est("estimator1")
        def fn1():
            pass

        @register_adv_est("estimator2")
        def fn2():
            pass

        self.assertEqual(len(self.ADV_ESTIMATOR_REGISTRY), 2 + init_adv_count)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator1"], fn1)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator2"], fn2)

    def test_get_adv_estimator_fn_valid_names(self):
        """Test that valid names return the correct function from registry."""
        # Test GAE
        gae_fn = get_adv_estimator_fn("gae")
        assert gae_fn(5) == 10  # 5 * 2 = 10

        # Test Vtrace
        vtrace_fn = get_adv_estimator_fn("vtrace")
        assert vtrace_fn(5) == 6  # 5 + 1 = 6

    def test_get_adv_estimator_fn_invalid_name(self):
        """Test that invalid names raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            get_adv_estimator_fn("invalid_name")
        assert "Unknown advantage estimator simply: invalid_name" in str(excinfo.value)

    def test_get_adv_estimator_fn_case_sensitive(self):
        """Test that name lookup is case-sensitive."""
        with pytest.raises(ValueError):
            get_adv_estimator_fn("GAE")  # Different case


def test_multi_turn_compute_gae_advantage_return():
    """Test multi-turn GAE skip observation tokens."""
    gamma = random.uniform(0.0, 1.0)
    lam = random.uniform(0.0, 1.0)

    rewards = torch.tensor([[0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0]], dtype=torch.float)

    values1 = torch.tensor(
        [
            [
                random.uniform(-100.0, 100.0),
                random.random(),
                4.0,
                5.0,
                6.0,
                random.uniform(-100.0, 0),
                random.random(),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    values2 = torch.tensor(
        [
            [
                random.random(),
                random.uniform(-100.0, 100.0),
                4.0,
                5.0,
                6.0,
                random.random(),
                random.uniform(0.0, 100.0),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    response_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.float)

    adv1, ret1 = compute_gae_advantage_return(rewards, values1, response_mask, gamma, lam)
    adv2, ret2 = compute_gae_advantage_return(rewards, values2, response_mask, gamma, lam)

    ret1 *= response_mask
    ret2 *= response_mask
    assert torch.equal(adv1, adv2), f"{adv1=}, {adv2=}"
    assert torch.equal(ret1, ret2), f"{ret1=}, {ret2=}"
    print(f" [CORRECT] \n\n{adv1=}, \n\n{ret1=}")


def _make_group_index(batch_size: int, num_groups: int) -> np.ndarray:
    """Create a numpy index array ensuring each group has at least 2 samples."""
    assert num_groups * 2 <= batch_size, "batch_size must allow >=2 samples per group"
    counts: list[int] = [2] * num_groups
    remaining = batch_size - 2 * num_groups
    for _ in range(remaining):
        counts[random.randrange(num_groups)] += 1
    index = []
    for gid, c in enumerate(counts):
        index.extend([gid] * c)
    random.shuffle(index)
    return np.asarray(index, dtype=np.int64)


def _rand_mask(batch_size: int, seq_len: int) -> torch.Tensor:
    mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.int64).float()
    rows_without_one = (mask.sum(dim=-1) == 0).nonzero(as_tuple=True)[0]
    if len(rows_without_one) > 0:
        mask[rows_without_one, -1] = 1.0
    return mask


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_rloo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    index = _make_group_index(batch_size, num_groups)
    response_mask = _rand_mask(batch_size, seq_len)
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask
    adv1, ret1 = compute_rloo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    adv2, ret2 = compute_rloo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    # Print concise diagnostics for visibility during test runs
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[RLOO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


def test_grpo_vectorized_matches_original_for_low_variance_rewards():
    token_level_rewards = torch.tensor([[1.0], [1.00001], [2.0], [2.00001]], dtype=torch.float32)
    response_mask = torch.ones_like(token_level_rewards)
    index = np.array(["prompt-a", "prompt-a", "prompt-b", "prompt-b"], dtype=object)

    adv1, ret1 = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    adv2, ret2 = compute_grpo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


def test_gradient_norm_grpo_uses_weighted_group_baseline():
    token_level_rewards = torch.tensor([[0.0], [2.0]], dtype=torch.float32)
    response_mask = torch.ones_like(token_level_rewards)
    index = np.array(["prompt-a", "prompt-a"], dtype=object)
    score_grad_norm_sq = torch.tensor([1.0, 3.0], dtype=torch.float64)

    advantages, returns = compute_grpo_gradient_norm_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        score_grad_norm_sq=score_grad_norm_sq,
        norm_adv_by_std_in_grpo=False,
    )

    # b* = (1 * 0 + 3 * 2) / (1 + 3) = 1.5
    expected = torch.tensor([[-1.5], [0.5]], dtype=torch.float32)
    assert torch.equal(advantages, expected)
    assert torch.equal(returns, expected)


def test_gradient_norm_grpo_equal_weights_matches_standard_grpo():
    token_level_rewards = torch.tensor([[1.0], [2.0], [5.0], [-1.0]], dtype=torch.float32)
    response_mask = torch.ones_like(token_level_rewards)
    index = np.array(["prompt-a", "prompt-a", "prompt-b", "prompt-b"], dtype=object)

    grpo_advantages, grpo_returns = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    weighted_advantages, weighted_returns = compute_grpo_gradient_norm_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        score_grad_norm_sq=torch.ones(4, dtype=torch.float64),
    )

    assert torch.allclose(weighted_advantages, grpo_advantages)
    assert torch.allclose(weighted_returns, grpo_returns)


def test_gradient_norm_baseline_minimizes_empirical_second_moment():
    rewards = torch.tensor([0.0, 1.0, 4.0], dtype=torch.float64)
    weights = torch.tensor([1.0, 2.0, 9.0], dtype=torch.float64)
    ordinary_baseline = rewards.mean()
    weighted_baseline = (weights * rewards).sum() / weights.sum()

    ordinary_second_moment = (weights * (rewards - ordinary_baseline).square()).sum()
    weighted_second_moment = (weights * (rewards - weighted_baseline).square()).sum()

    assert weighted_second_moment < ordinary_second_moment


@pytest.mark.parametrize(
    "invalid_weights,error_match",
    [
        (torch.tensor([1.0]), "one scalar per response"),
        (torch.tensor([1.0, -1.0]), "non-negative"),
        (torch.tensor([1.0, torch.inf]), "NaN or infinity"),
    ],
)
@pytest.mark.parametrize(
    "advantage_fn",
    [compute_grpo_gradient_norm_outcome_advantage, compute_grpo_gradient_norm_loo_outcome_advantage],
)
def test_gradient_norm_grpo_rejects_invalid_weights(invalid_weights: torch.Tensor, error_match: str, advantage_fn):
    with pytest.raises(ValueError, match=error_match):
        advantage_fn(
            token_level_rewards=torch.tensor([[0.0], [1.0]]),
            response_mask=torch.ones(2, 1),
            index=np.array(["prompt-a", "prompt-a"], dtype=object),
            score_grad_norm_sq=invalid_weights,
        )


def test_grpo_loo_excludes_current_response_reward():
    token_level_rewards = torch.tensor([[0.0], [2.0], [4.0]], dtype=torch.float32)
    response_mask = torch.ones_like(token_level_rewards)
    index = np.array(["prompt-a"] * 3, dtype=object)

    advantages, returns = compute_grpo_loo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        norm_adv_by_std_in_grpo=False,
    )

    # LOO baselines are [3, 2, 1], respectively.
    expected = torch.tensor([[-3.0], [0.0], [3.0]], dtype=torch.float32)
    assert torch.equal(advantages, expected)
    assert torch.equal(returns, expected)


def test_gradient_norm_grpo_loo_uses_only_other_responses():
    token_level_rewards = torch.tensor([[0.0], [2.0], [4.0]], dtype=torch.float32)
    response_mask = torch.ones_like(token_level_rewards)
    index = np.array(["prompt-a"] * 3, dtype=object)
    score_grad_norm_sq = torch.tensor([1.0, 3.0, 6.0], dtype=torch.float64)

    advantages, returns = compute_grpo_gradient_norm_loo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        score_grad_norm_sq=score_grad_norm_sq,
        norm_adv_by_std_in_grpo=False,
    )

    expected = torch.tensor([[-10.0 / 3.0], [-10.0 / 7.0], [2.5]], dtype=torch.float32)
    assert torch.allclose(advantages, expected)
    assert torch.allclose(returns, expected)


def test_gradient_norm_grpo_loo_equal_weights_matches_grpo_loo():
    token_level_rewards = torch.tensor([[1.0], [2.0], [5.0], [-1.0]], dtype=torch.float32)
    response_mask = torch.ones_like(token_level_rewards)
    index = np.array(["prompt-a", "prompt-a", "prompt-b", "prompt-b"], dtype=object)

    loo_advantages, loo_returns = compute_grpo_loo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    weighted_advantages, weighted_returns = compute_grpo_gradient_norm_loo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        score_grad_norm_sq=torch.ones(4, dtype=torch.float64),
    )

    assert torch.allclose(weighted_advantages, loo_advantages)
    assert torch.allclose(weighted_returns, loo_returns)


def test_gradient_norm_grpo_loo_zero_held_out_weight_falls_back_to_ordinary_loo():
    rewards = torch.tensor([[1.0], [5.0]], dtype=torch.float32)
    mask = torch.ones_like(rewards)
    index = np.array(["prompt-a", "prompt-a"], dtype=object)

    ordinary, _ = compute_grpo_loo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        norm_adv_by_std_in_grpo=False,
    )
    weighted, _ = compute_grpo_gradient_norm_loo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        score_grad_norm_sq=torch.tensor([1.0, 0.0]),
        norm_adv_by_std_in_grpo=False,
    )

    assert torch.equal(weighted, ordinary)


def test_gradient_norm_grpo_loo_baseline_is_independent_of_current_pair():
    rewards = torch.tensor([[1.0], [3.0], [7.0]], dtype=torch.float32)
    mask = torch.ones_like(rewards)
    index = np.array(["prompt-a"] * 3, dtype=object)

    original, _ = compute_grpo_gradient_norm_loo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        score_grad_norm_sq=torch.tensor([2.0, 5.0, 11.0]),
        norm_adv_by_std_in_grpo=False,
    )
    perturbed, _ = compute_grpo_gradient_norm_loo_outcome_advantage(
        token_level_rewards=torch.tensor([[101.0], [3.0], [7.0]]),
        response_mask=mask,
        index=index,
        score_grad_norm_sq=torch.tensor([2000.0, 5.0, 11.0]),
        norm_adv_by_std_in_grpo=False,
    )

    original_baseline = rewards[0, 0] - original[0, 0]
    perturbed_baseline = 101.0 - perturbed[0, 0]
    assert torch.allclose(original_baseline, perturbed_baseline)


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_grpo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Generate group indices (numpy array of shape [batch_size])
    index = _make_group_index(batch_size, num_groups)

    # Generate binary response mask (at least one valid token per row)
    response_mask = _rand_mask(batch_size, seq_len)

    # Generate token-level rewards and apply mask
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask

    # Compute GRPO outcome advantage (original implementation)
    adv1, ret1 = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Compute GRPO outcome advantage (vectorized implementation)
    adv2, ret2 = compute_grpo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Diagnostic info for visibility (same style as RLOO test)
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[GRPO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )

    # Assert shape and numerical equivalence
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "name,base",
    [
        ("k1+", "k1"),
        ("kl+", "kl"),
        ("abs+", "abs"),
        ("k3+", "k3"),
        ("low_var_kl+", "low_var_kl"),
    ],
)
def test_kl_penalty_straight_through_value_matches_base(name, base):
    """The ``+`` suffix is a straight-through trick that swaps in the k2
    gradient while keeping the base estimator's value. Therefore the forward
    value of e.g. ``k3+`` must match the value of plain ``k3``.

    Regression test for the bug where ``kl_penalty(..., "k3+")`` raised
    ``NotImplementedError`` because the wrapper forwarded the ``+`` suffix to
    ``kl_penalty_forward`` without stripping it.
    """
    torch.manual_seed(0)
    logprob = torch.randn(4, 8, requires_grad=True)
    ref_logprob = torch.randn(4, 8)

    plus_value = kl_penalty(logprob, ref_logprob, name)
    base_value = kl_penalty(logprob, ref_logprob, base)
    assert torch.allclose(plus_value, base_value)


def test_kl_penalty_k3_plus_uses_k2_gradient():
    """With ``k3+`` the gradient w.r.t. ``logprob`` should equal the gradient
    obtained from the ``k2`` (``0.5 * log_ratio**2``) estimator, since the
    straight-through trick routes the backward pass through ``k2``.
    """
    torch.manual_seed(0)
    logprob = torch.randn(4, 8, requires_grad=True)
    ref_logprob = torch.randn(4, 8)

    out_plus = kl_penalty(logprob, ref_logprob, "k3+").sum()
    (grad_plus,) = torch.autograd.grad(out_plus, logprob)

    logprob_k2 = logprob.detach().clone().requires_grad_(True)
    out_k2 = kl_penalty(logprob_k2, ref_logprob, "k2").sum()
    (grad_k2,) = torch.autograd.grad(out_k2, logprob_k2)

    assert torch.allclose(grad_plus, grad_k2)


if __name__ == "__main__":
    unittest.main()
