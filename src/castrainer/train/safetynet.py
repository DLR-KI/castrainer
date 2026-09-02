# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""SafetyNet implementation."""

import torch

from castrainer.train.data import CASDataset
from castrainer.train.lut import KDTreeLUT
from castrainer.train.net import Net


class SafetyNet:
    """SafetyNet class.

    Args:
        model (Net): The model to be used for prediction.
    """

    def __init__(self, model: Net) -> None:
        """Wrap a trained model with an initially empty LUT.

        Args:
            model (Net): The model to be used for prediction.
        """
        self.model = model
        self.lut = KDTreeLUT.empty()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the output for the given input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        self.model.eval()
        y = torch.empty(
            (x.shape[0], self.model.outputs), dtype=torch.float32, device=x.device
        )

        lut_mask_values: list[bool] = []
        lut_indices: list[int] = []
        lut_values: list[torch.Tensor] = []
        for index, sample in enumerate(x):
            if sample in self.lut:
                lut_mask_values.append(True)
                lut_indices.append(index)
                lut_values.append(self.lut[sample].to(device=x.device))
            else:
                lut_mask_values.append(False)

        lut_mask = torch.tensor(lut_mask_values, dtype=torch.bool, device=x.device)
        if lut_values:
            y[lut_indices] = torch.stack(lut_values)

        if (~lut_mask).any():
            y[~lut_mask] = self.model.encoder(x[~lut_mask])

        return y

    def fill(
        self,
        dataset: CASDataset | None = None,
        x: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> None:
        """Fill the lookup table with wrongly predicted values.

        Args:
            dataset (CASDataset | None): The dataset to be used for
                filling the LUT.
            x (torch.Tensor | None): The input tensor.
            y (torch.Tensor | None): The target tensor.

        Raises:
            ValueError: If dataset or X and y are not provided.
        """
        if (x is None or y is None) and dataset is None:
            raise ValueError("Either dataset or X and y must be provided")
        if dataset is not None:
            x = dataset.x
            y = dataset.y
        if x is None or y is None:
            raise ValueError("Something went wrong")
        self.model.eval()
        y_hat: torch.Tensor = self.model.encoder(x)

        # Save wrongly predicted values to the LUT
        lut_entries: list[tuple[torch.Tensor, torch.Tensor]] = []
        for x, y_true, y_pred in zip(x, y, y_hat, strict=False):  # ruff: ignore[loop-variable-overrides-iterator]
            if y_true.argmax() != y_pred.argmax():
                lut_entries.append((x.detach().clone(), y_true.detach().clone()))

        self.lut = KDTreeLUT.from_items(lut_entries)
