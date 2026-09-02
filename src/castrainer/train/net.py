# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""The neural network for the castrainer package."""

import lightning as L  # ruff: ignore[lowercase-imported-as-non-lowercase]
import torch
from torch import nn, optim


class Net(L.LightningModule):
    """Simple feedforward neural network.

    Args:
        inputs (int): Number of input features.
        outputs (int): Number of output features.
        hidden_layers (list[int] | None): Number of nodes in each
            hidden layer.
        activation (type): Activation function.
        one_hot (bool): Whether to use one-hot encoding for
            the target.
    """

    def __init__(
        self,
        *,
        inputs: int = 3,
        outputs: int = 5,
        hidden_layers: list[int] | None = None,
        activation: type = nn.ReLU,
        one_hot: bool = False,
    ) -> None:
        """Build the feedforward encoder from the given layer sizes.

        Args:
            inputs (int): Number of input features.
            outputs (int): Number of output features.
            hidden_layers (list[int] | None): Number of nodes in each
                hidden layer.
            activation (type): Activation function.
            one_hot (bool): Whether to use one-hot encoding for the
                target.
        """
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.hidden_layers: list[int] = (
            hidden_layers if hidden_layers is not None else [5, 5, 5, 5, 5]
        )
        self.activation = activation
        self.one_hot = one_hot

        layers = []
        in_size = inputs
        for out_size in self.hidden_layers:
            layers.extend((nn.Linear(in_size, out_size), self.activation()))
            in_size = out_size
        layers.append(nn.Linear(in_size, outputs))
        self.encoder = nn.Sequential(*layers)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,  # ruff: ignore[unused-method-argument]
    ) -> torch.Tensor:
        """The training step.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): The batch of
                data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss.
        """
        x, y = batch
        y_pred = self.encoder(x)
        if self.one_hot:
            loss = nn.functional.cross_entropy(y_pred, y.argmax(dim=1))
        else:
            loss = nn.functional.mse_loss(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,  # ruff: ignore[unused-method-argument]
    ) -> torch.Tensor:
        """The test step.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): The batch of
                data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss.
        """
        x, y = batch
        y_pred = self.encoder(x)
        if self.one_hot:
            loss = nn.functional.cross_entropy(y_pred, y.argmax(dim=1))
        else:
            loss = nn.functional.mse_loss(y_pred, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        return optim.Adam(self.parameters(), lr=1e-3)
