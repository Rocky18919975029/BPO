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

import numpy as np
import pytest
import torch

from verl.trainer.ppo.core_algos import compute_policy_loss_positive_sft, compute_positive_sft_weights


def test_positive_sft_weights_equalize_active_prompts():
    rewards = torch.tensor([[1.0], [-1.0], [1.0], [-1.0], [1.0]])
    mask = torch.ones_like(rewards)
    prompt_ids = np.array(["a", "a", "a", "b", "b"], dtype=object)

    weights, returns = compute_positive_sft_weights(
        token_level_rewards=rewards,
        response_mask=mask,
        index=prompt_ids,
        config={"positive_sft_reward_threshold": 0.0},
    )

    # Three retained responses and two active prompts. Prompt a has two
    # positives (0.75 each); prompt b has one positive (1.5).
    expected = torch.tensor([[0.75], [0.0], [0.75], [0.0], [1.5]])
    torch.testing.assert_close(weights, expected)
    torch.testing.assert_close(returns, expected)
    assert weights[torch.tensor([0, 2])].sum().item() == pytest.approx(weights[4].sum().item())
    assert weights[weights > 0].mean().item() == pytest.approx(1.0)


def test_positive_sft_weights_skip_groups_without_correct_response():
    rewards = torch.tensor([[-1.0], [-1.0], [1.0], [-1.0]])
    mask = torch.ones_like(rewards)
    prompt_ids = np.array(["a", "a", "b", "b"], dtype=object)

    weights, _ = compute_positive_sft_weights(
        token_level_rewards=rewards,
        response_mask=mask,
        index=prompt_ids,
    )

    torch.testing.assert_close(weights, torch.tensor([[0.0], [0.0], [1.0], [0.0]]))


def test_positive_sft_weights_all_incorrect_are_zero():
    rewards = -torch.ones(4, 2)
    mask = torch.ones_like(rewards)
    prompt_ids = np.array(["a", "a", "b", "b"], dtype=object)

    weights, _ = compute_positive_sft_weights(
        token_level_rewards=rewards,
        response_mask=mask,
        index=prompt_ids,
    )

    assert torch.count_nonzero(weights) == 0


def test_positive_sft_loss_is_weighted_response_mean_log_likelihood():
    log_prob = torch.tensor([[-1.0, -3.0], [-2.0, -4.0]])
    old_log_prob = log_prob.clone()
    response_mask = torch.ones_like(log_prob, dtype=torch.bool)
    # Response means are 2 and 3. Their problem-balanced weights are 0.5 and
    # 1.5, so the sequence-mean loss is (0.5*2 + 1.5*3) / 2 = 2.75.
    weights = torch.tensor([[0.5, 0.5], [1.5, 1.5]])

    loss, metrics = compute_policy_loss_positive_sft(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=weights,
        response_mask=response_mask,
        loss_agg_mode="seq-mean-token-mean",
    )

    assert loss.item() == pytest.approx(2.75)
    assert metrics["positive_sft/token_weight_mean"] == pytest.approx(1.0)
    assert metrics["positive_sft/approx_kl"] == pytest.approx(0.0)


def test_positive_sft_loss_rejects_non_response_balanced_aggregation():
    values = torch.zeros(1, 1)
    with pytest.raises(ValueError, match="seq-mean-token-mean"):
        compute_policy_loss_positive_sft(
            old_log_prob=values,
            log_prob=values,
            advantages=torch.ones_like(values),
            response_mask=torch.ones_like(values, dtype=torch.bool),
            loss_agg_mode="token-mean",
        )
