# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for GPU-memory and config-loading helpers in the study CLI."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from castrainer.cli.study import (
    _gpu_meets_memory_threshold,
    check_gpu_memory,
    load_hyperparameter_config,
)


def test_load_hyperparameter_config_missing_file() -> None:
    with pytest.raises(ValueError, match="Config file not found"):
        load_hyperparameter_config("nonexistent.yaml")


def test_load_hyperparameter_config_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("activations:\n  - relu\n  - gelu\n")
    config = load_hyperparameter_config(str(config_path))
    assert config == {"activations": ["relu", "gelu"]}


def test_load_hyperparameter_config_json(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    with config_path.open("w") as f:
        json.dump({"activations": ["tanh"]}, f)
    config = load_hyperparameter_config(str(config_path))
    assert config == {"activations": ["tanh"]}


def test_gpu_meets_memory_threshold_true() -> None:
    mock_info = MagicMock(free=4 * 1024**3)
    with (
        patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=MagicMock()),
        patch("pynvml.nvmlDeviceGetMemoryInfo", return_value=mock_info),
    ):
        assert _gpu_meets_memory_threshold(0, min_free_memory_gb=2.0) is True


def test_gpu_meets_memory_threshold_false() -> None:
    mock_info = MagicMock(free=1 * 1024**3)
    with (
        patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=MagicMock()),
        patch("pynvml.nvmlDeviceGetMemoryInfo", return_value=mock_info),
    ):
        assert _gpu_meets_memory_threshold(0, min_free_memory_gb=2.0) is False


def test_check_gpu_memory_all_sufficient() -> None:
    mock_info = MagicMock(free=8 * 1024**3)
    with (
        patch("pynvml.nvmlInit"),
        patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=MagicMock()),
        patch("pynvml.nvmlDeviceGetMemoryInfo", return_value=mock_info),
    ):
        assert check_gpu_memory([0, 1], min_free_memory_gb=2.0) is True


def test_check_gpu_memory_one_insufficient() -> None:
    infos = [MagicMock(free=8 * 1024**3), MagicMock(free=1 * 1024**3)]
    with (
        patch("pynvml.nvmlInit"),
        patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=MagicMock()),
        patch("pynvml.nvmlDeviceGetMemoryInfo", side_effect=infos),
    ):
        assert check_gpu_memory([0, 1], min_free_memory_gb=2.0) is False


def test_check_gpu_memory_import_error() -> None:
    with patch("pynvml.nvmlInit", side_effect=ImportError):
        assert check_gpu_memory([0], min_free_memory_gb=2.0) is False


def test_check_gpu_memory_unexpected_error() -> None:
    with patch("pynvml.nvmlInit", side_effect=RuntimeError("boom")):
        assert check_gpu_memory([0], min_free_memory_gb=2.0) is False
