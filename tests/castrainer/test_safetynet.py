# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for the SafetyNet runtime behavior."""

from typing import Any, cast

import pytest
import torch
from torch import nn

from castrainer.train.net import Net
from castrainer.train.safetynet import SafetyNet


def test_safetynet_fill_and_predict_uses_kdtree_lut() -> None:
    """SafetyNet should store mispredictions in a k-d-tree LUT."""
    model = Net(inputs=2, outputs=2, hidden_layers=[], activation=nn.ReLU)
    cast("Any", model).encoder = nn.Identity()

    safetynet = SafetyNet(model)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    y = torch.tensor([[0.0, 0.0], [3.0, 4.0]], dtype=torch.float32)

    safetynet.fill(x=x, y=y)

    assert len(safetynet.lut) == 1
    assert safetynet.lut[torch.tensor([1.0, 2.0])].shape == torch.Size([2])

    prediction = safetynet.predict(x)

    assert torch.equal(prediction[0], y[0])
    assert torch.equal(prediction[1], x[1])


def test_safetynet_fill_requires_dataset_or_x_and_y() -> None:
    model = Net(inputs=2, outputs=2, hidden_layers=[])
    safetynet = SafetyNet(model)

    with pytest.raises(ValueError, match="Either dataset or X and y"):
        safetynet.fill()


def test_safetynet_fill_requires_both_x_and_y() -> None:
    model = Net(inputs=2, outputs=2, hidden_layers=[])
    safetynet = SafetyNet(model)

    with pytest.raises(ValueError, match="Either dataset or X and y"):
        safetynet.fill(x=torch.tensor([[1.0, 2.0]]))


def test_safetynet_fill_rejects_dataset_with_missing_x() -> None:
    # A dataset-like object whose x/y attributes are None is caught by
    # the defensive second check, distinct from the "nothing provided
    # at all" case above.
    model = Net(inputs=2, outputs=2, hidden_layers=[])
    safetynet = SafetyNet(model)

    class _BrokenDataset:
        x = None
        y = torch.tensor([[1.0, 2.0]])

    with pytest.raises(ValueError, match="Something went wrong"):
        safetynet.fill(dataset=cast("Any", _BrokenDataset()))
