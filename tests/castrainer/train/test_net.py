# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for the Net LightningModule."""

import pytest
import torch
from torch import nn, optim

from castrainer.train.net import Net


def test_net_default_hidden_layers() -> None:
    model = Net()
    assert model.hidden_layers == [5, 5, 5, 5, 5]
    assert model.inputs == 3
    assert model.outputs == 5
    assert model.one_hot is False


def test_net_custom_layers_build_encoder() -> None:
    model = Net(inputs=4, outputs=2, hidden_layers=[8, 6], activation=nn.ReLU)
    linear_layers = [m for m in model.encoder if isinstance(m, nn.Linear)]
    assert [layer.in_features for layer in linear_layers] == [4, 8, 6]
    assert [layer.out_features for layer in linear_layers] == [8, 6, 2]


def test_net_training_step_returns_mse_loss() -> None:
    model = Net(inputs=2, outputs=2, hidden_layers=[3])
    x = torch.randn(4, 2)
    y = torch.randn(4, 2)

    loss = model.training_step((x, y), 0)

    assert loss.ndim == 0
    assert loss.item() >= 0


def test_net_training_step_cross_entropy_when_one_hot() -> None:
    model = Net(inputs=2, outputs=3, hidden_layers=[3], one_hot=True)
    x = torch.randn(4, 2)
    y = nn.functional.one_hot(torch.randint(0, 3, (4,)), num_classes=3).float()

    loss = model.training_step((x, y), 0)

    y_pred = model.encoder(x)
    expected = nn.functional.cross_entropy(y_pred, y.argmax(dim=1))
    assert torch.isclose(loss, expected)


def test_net_test_step_mse_when_not_one_hot() -> None:
    model = Net(inputs=2, outputs=2, hidden_layers=[3], one_hot=False)
    x = torch.randn(4, 2)
    y = torch.randn(4, 2)

    loss = model.test_step((x, y), 0)

    y_pred = model.encoder(x)
    expected = nn.functional.mse_loss(y_pred, y)
    assert torch.isclose(loss, expected)


def test_net_test_step_cross_entropy_when_one_hot() -> None:
    model = Net(inputs=2, outputs=3, hidden_layers=[3], one_hot=True)
    x = torch.randn(4, 2)
    y = nn.functional.one_hot(torch.randint(0, 3, (4,)), num_classes=3).float()

    loss = model.test_step((x, y), 0)

    y_pred = model.encoder(x)
    expected = nn.functional.cross_entropy(y_pred, y.argmax(dim=1))
    assert torch.isclose(loss, expected)


def test_net_configure_optimizers_returns_adam() -> None:
    model = Net(inputs=2, outputs=2, hidden_layers=[3])
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, optim.Adam)
    assert optimizer.defaults["lr"] == pytest.approx(1e-3)
