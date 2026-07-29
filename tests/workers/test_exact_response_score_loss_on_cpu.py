# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.utils.losses import exact_response_score_loss


def _make_data(response_mask: list[int]) -> TensorDict:
    return TensorDict(
        {
            "prompts": torch.tensor([[10, 11]]),
            "responses": torch.tensor([[12, 13, 14]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "response_mask": torch.tensor([response_mask], dtype=torch.float32),
        },
        batch_size=[1],
    )


def _make_log_probs() -> torch.Tensor:
    # no_padding_2_padding left-shifts and extracts positions 1:4. The
    # response mask then keeps values -1 and -2, whose negative sum is 3.
    return torch.tensor([-99.0, -1.0, -2.0, -3.0, -99.0], requires_grad=True)


@pytest.mark.parametrize(
    "loss_agg_mode,expected",
    [
        ("token-mean", 3.0),
        ("seq-mean-token-sum", 3.0),
        ("seq-mean-token-mean", 1.5),
    ],
)
def test_exact_response_score_loss_matches_actor_aggregation(loss_agg_mode: str, expected: float):
    data = _make_data([1, 1, 0])
    tu.assign_non_tensor_data(data, "score_loss_agg_mode", loss_agg_mode)

    loss, metrics = exact_response_score_loss({"log_probs": _make_log_probs()}, data)

    assert metrics == {}
    assert torch.allclose(loss, torch.tensor(expected))


def test_exact_response_score_loss_uses_configured_sum_normalization():
    data = _make_data([1, 1, 0])
    tu.assign_non_tensor_data(data, "score_loss_agg_mode", "seq-mean-token-sum-norm")
    tu.assign_non_tensor_data(data, "score_loss_scale_factor", 4.0)

    loss, _ = exact_response_score_loss({"log_probs": _make_log_probs()}, data)

    assert torch.allclose(loss, torch.tensor(0.75))


def test_exact_response_score_loss_gives_padding_rows_zero_gradient():
    data = _make_data([0, 0, 0])
    tu.assign_non_tensor_data(data, "score_loss_agg_mode", "seq-mean-token-mean")
    log_probs = _make_log_probs()

    loss, _ = exact_response_score_loss({"log_probs": log_probs}, data)
    loss.backward()

    assert loss.item() == 0.0
    assert torch.count_nonzero(log_probs.grad).item() == 0
