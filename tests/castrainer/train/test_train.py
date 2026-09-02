# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for the sequential training loop and GPU status helpers."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
from torch import nn
from torch.utils.data import DataLoader

from castrainer.train.data import CASDataset
from castrainer.train.net import Net
from castrainer.train.train import _query_gpu_status, train


def _write_h5(path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.random((8, 3), dtype=np.float32)
    y = rng.random((8, 2), dtype=np.float32)
    with h5py.File(str(path), "w") as f:
        f.create_dataset("X", data=x)
        f.create_dataset("y", data=y)


def test_train_invalid_strategy_raises() -> None:
    with pytest.raises(ValueError, match="Invalid strategy"):
        train(
            models=[],
            train_dataloader=None,  # ty: ignore[invalid-argument-type]
            test_dataloader=None,  # ty: ignore[invalid-argument-type]
            trainer_kwargs={},
            strategy="bogus",  # ty: ignore[invalid-argument-type]
        )


def test_train_sequential_writes_results_and_training_info(tmp_path: Path) -> None:
    h5_path = tmp_path / "data.h5"
    _write_h5(h5_path)
    dataset = CASDataset(h5_path)

    train_dataloader = DataLoader(dataset, batch_size=4, num_workers=0)
    test_dataloader = DataLoader(dataset, batch_size=4, num_workers=0)

    model = Net(inputs=3, outputs=2, hidden_layers=[4], activation=nn.ReLU)

    output_dir = tmp_path / "output"
    train(
        models=[model],
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        trainer_kwargs={
            "max_epochs": 1,
            "enable_progress_bar": False,
            "enable_checkpointing": False,
            "logger": False,
        },
        strategy="sequential",
        output_dir=output_dir,
    )

    results_files = list(tmp_path.glob("data_results-*.json"))
    assert len(results_files) == 1
    with results_files[0].open("r", encoding="utf-8") as f:
        results = json.load(f)
    assert len(results) == 1
    assert results[0]["inputs"] == 3
    assert results[0]["outputs"] == 2

    info_file = output_dir / "training_info.json"
    assert info_file.exists()
    with info_file.open("r", encoding="utf-8") as f:
        info = json.load(f)
    assert info["hyperparameters"]["inputs"] == 3
    assert info["dataset"]["size"] == 8


def test_query_gpu_status_success() -> None:
    mock_handle = MagicMock()
    mock_info = MagicMock(free=4 * 1024**3, total=8 * 1024**3)

    with (
        patch(
            "castrainer.train.train.pynvml.nvmlDeviceGetHandleByIndex",
            return_value=mock_handle,
        ),
        patch(
            "castrainer.train.train.pynvml.nvmlDeviceGetMemoryInfo",
            return_value=mock_info,
        ),
    ):
        result = _query_gpu_status(0)

    assert result is not None
    dev_idx, free_gb, total_gb = result
    assert dev_idx == 0
    assert free_gb == pytest.approx(4.0)
    assert total_gb == pytest.approx(8.0)


def test_query_gpu_status_nvml_error_returns_none() -> None:
    import pynvml

    with patch(
        "castrainer.train.train.pynvml.nvmlDeviceGetHandleByIndex",
        side_effect=pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN),
    ):
        assert _query_gpu_status(0) is None
