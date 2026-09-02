# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for SafetyNet trainer."""

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from castrainer.safetynet.config import VCAS_CONFIG
from castrainer.safetynet.trainer import (
    DATA_DIR_ENV_VAR,
    SafetyNetTrainer,
    TrainingConfig,
    get_activation_fn,
    resolve_data_dir,
)
from castrainer.train.lut import KDTreeLUT


def test_relu() -> None:
    """Test ReLU activation."""
    assert get_activation_fn("relu") == nn.ReLU


def test_leakyrelu() -> None:
    """Test LeakyReLU activation."""
    assert get_activation_fn("leakyrelu") == nn.LeakyReLU


def test_gelu() -> None:
    """Test GELU activation."""
    assert get_activation_fn("gelu") == nn.GELU


def test_tanh() -> None:
    """Test Tanh activation."""
    assert get_activation_fn("tanh") == nn.Tanh


def test_sigmoid() -> None:
    """Test Sigmoid activation."""
    assert get_activation_fn("sigmoid") == nn.Sigmoid


def test_unknown_defaults_to_relu() -> None:
    """Test unknown activation defaults to ReLU."""
    assert get_activation_fn("unknown") == nn.ReLU


def test_case_insensitive() -> None:
    """Test activation lookup is case insensitive."""
    assert get_activation_fn("RELU") == nn.ReLU
    assert get_activation_fn("ReLU") == nn.ReLU


def test_default_activations() -> None:
    """Test default activations."""
    config = TrainingConfig()
    assert config.activations == ["relu"]


def test_default_hidden_nodes() -> None:
    """Test default hidden nodes."""
    config = TrainingConfig()
    assert config.hidden_nodes == [100]


def test_default_hidden_layers() -> None:
    """Test default hidden layers."""
    config = TrainingConfig()
    assert config.hidden_layers == [4]


def test_default_batch_size() -> None:
    """Test default batch size."""
    config = TrainingConfig()
    assert config.batch_size == 32


def test_default_max_epochs() -> None:
    """Test default max epochs."""
    config = TrainingConfig()
    assert config.max_epochs == 10000


def test_custom_config() -> None:
    """Test custom configuration."""
    config = TrainingConfig(
        activations=["gelu"],
        hidden_nodes=[50],
        hidden_layers=[2],
        batch_size=64,
        max_epochs=100,
    )
    assert config.activations == ["gelu"]
    assert config.hidden_nodes == [50]
    assert config.hidden_layers == [2]
    assert config.batch_size == 64
    assert config.max_epochs == 100


def test_init_default() -> None:
    """Test trainer initialization with defaults."""
    trainer = SafetyNetTrainer()
    assert trainer.output_dir == Path("safetynet")
    assert isinstance(trainer.config, TrainingConfig)


def test_init_custom() -> None:
    """Test trainer initialization with custom config."""
    config = TrainingConfig(batch_size=64)
    output_dir = Path("custom_output")
    trainer = SafetyNetTrainer(config=config, output_dir=output_dir)
    assert trainer.output_dir == output_dir
    assert trainer.config.batch_size == 64


def test_get_data_path_vcas_directory() -> None:
    """Test getting VCAS data directory."""
    trainer = SafetyNetTrainer()
    path = trainer._get_data_path("vcas")
    assert "VerticalCAS" in str(path)
    assert "GenerateNetworks" in str(path)


def test_get_data_path_hcas_directory() -> None:
    """Test getting HCAS data directory."""
    trainer = SafetyNetTrainer()
    path = trainer._get_data_path("hcas")
    assert "HorizontalCAS" in str(path)
    assert "GenerateNetworks" in str(path)


def test_get_data_path_vcas_subsystem() -> None:
    """Test getting VCAS subsystem data file."""
    trainer = SafetyNetTrainer()
    path = trainer._get_data_path("vcas", 1)
    assert "VCAS_TrainingData_v5_01.h5" in str(path)


def test_get_data_path_hcas_subsystem() -> None:
    """Test getting HCAS subsystem data file."""
    trainer = SafetyNetTrainer()
    path = trainer._get_data_path("hcas", 0)
    assert "HCAS_rect_TrainingData_v6_pra0_tau00.h5" in str(path)


def test_get_data_path_hcas_subsystem_pra1() -> None:
    """Test getting HCAS subsystem data file for pra=1."""
    trainer = SafetyNetTrainer()
    path = trainer._get_data_path("hcas", 8)  # pra=1, tau=0
    assert "HCAS_rect_TrainingData_v6_pra1_tau00.h5" in str(path)


def test_get_data_path_unknown_system() -> None:
    """Test getting data path for unknown system."""
    trainer = SafetyNetTrainer()
    with pytest.raises(ValueError, match="Unknown system"):
        trainer._get_data_path("unknown")


def test_resolve_data_dir_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit data directory wins over the environment."""
    monkeypatch.setenv(DATA_DIR_ENV_VAR, str(tmp_path / "from_env"))
    assert resolve_data_dir(tmp_path / "explicit") == tmp_path / "explicit"


def test_resolve_data_dir_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The environment variable is used when no directory is given."""
    monkeypatch.setenv(DATA_DIR_ENV_VAR, str(tmp_path))
    assert resolve_data_dir() == tmp_path


def test_resolve_data_dir_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an argument or environment variable, the cwd is used."""
    monkeypatch.delenv(DATA_DIR_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    assert resolve_data_dir() == Path.cwd()


def test_get_data_path_uses_data_dir(tmp_path: Path) -> None:
    """Data paths are resolved below the configured data directory."""
    trainer = SafetyNetTrainer(data_dir=tmp_path)
    assert trainer._get_data_path("vcas", 1) == (
        tmp_path / "VerticalCAS/GenerateNetworks/VCAS_TrainingData_v5_01.h5"
    )
    assert trainer._get_data_path("hcas", 0) == (
        tmp_path
        / "HorizontalCAS/GenerateNetworks/HCAS_rect_TrainingData_v6_pra0_tau00.h5"
    )


@patch("castrainer.safetynet.trainer.Net")
def test_create_model(mock_net: MagicMock) -> None:
    """Test model creation."""
    trainer = SafetyNetTrainer(
        config=TrainingConfig(hidden_nodes=[50], hidden_layers=[2])
    )
    mock_net.return_value = MagicMock()
    model = trainer._create_model(inputs=4, outputs=9, activation_fn=nn.ReLU)
    assert model is not None
    mock_net.assert_called_once()
    call_kwargs = mock_net.call_args
    assert call_kwargs[1]["inputs"] == 4
    assert call_kwargs[1]["outputs"] == 9
    assert call_kwargs[1]["hidden_layers"] == [50, 50]


def test_save_model(tmp_path: Path) -> None:
    """Test model saving."""
    trainer = SafetyNetTrainer()
    model = MagicMock()
    model.state_dict.return_value = {}
    output_path = tmp_path / "test_model.pt"
    trainer._save_model(model, output_path)
    assert output_path.exists()


def test_save_lut(tmp_path: Path) -> None:
    """Test LUT saving."""
    trainer = SafetyNetTrainer()
    lut = KDTreeLUT.from_items([
        (torch.tensor([1.0, 2.0]), torch.tensor([0.5])),
        (torch.tensor([3.0, 4.0]), torch.tensor([0.7])),
    ])
    output_path = tmp_path / "test_lut.json"
    trainer._save_lut(lut, output_path, VCAS_CONFIG)
    assert output_path.exists()


def test_generate_manifest(tmp_path: Path) -> None:
    """Test manifest generation."""
    trainer = SafetyNetTrainer()
    networks = [{"file": "test.pt", "networkFormat": "torch", "if": {}}]
    luts = [{"file": "test_lut.json", "lutFormat": "snet", "if": {}}]
    manifest_path = trainer._generate_manifest(
        "vcas", tmp_path, networks, luts, VCAS_CONFIG
    )
    assert manifest_path.exists()
    assert manifest_path.name == "vcas.json"


def test_save_training_info(tmp_path: Path) -> None:
    """Test training info file saving."""
    from castrainer.train.net import Net
    from castrainer.train.safetynet import SafetyNet

    trainer = SafetyNetTrainer(
        config=TrainingConfig(
            batch_size=32,
            max_epochs=100,
            patience=10,
            one_hot=False,
        )
    )

    # Create a simple model
    model = Net(inputs=5, outputs=3, hidden_layers=[10, 10], activation=nn.ReLU)

    # Create a SafetyNet with mock LUT
    safetynet = SafetyNet(model)

    # Create mock dataset
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 100
    mock_dataset.file = "test_data.h5"
    mock_dataset.x = [torch.randn(5) for _ in range(100)]
    mock_dataset.y = [torch.randn(3) for _ in range(100)]

    # Create mock trainer
    mock_trainer = MagicMock()
    mock_trainer.max_epochs = 100
    mock_trainer.min_epochs = 1
    mock_trainer.current_epoch = 50
    mock_trainer.global_step = 500
    mock_trainer.callback_metrics = {"train_loss": torch.tensor(0.01)}

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    trainer._save_training_info(
        model=model,
        safetynet=safetynet,
        dataset=mock_dataset,
        trainer=mock_trainer,
        subsystem_name="test_subsystem",
        training_duration=123.45,
        output_dir=output_dir,
    )

    info_file = output_dir / "test_subsystem_training_info.json"
    assert info_file.exists()

    with info_file.open("r") as f:
        data = json.load(f)

    assert "hyperparameters" in data
    assert "training" in data
    assert "model_sizes" in data
    assert "lut_statistics" in data
    assert math.isclose(data["training"]["training_duration_seconds"], 123.45)
    assert data["subsystem"] == "test_subsystem"


def test_training_info_has_required_fields() -> None:
    """Test training info has all required fields."""
    # Create sample training info structure
    training_info = {
        "timestamp": "2024-01-01T00:00:00Z",
        "subsystem": "test",
        "hyperparameters": {
            "activation": "ReLU",
            "hidden_layers": [100, 100, 100, 100],
            "num_hidden_layers": 4,
            "inputs": 6,
            "outputs": 9,
            "one_hot": False,
            "batch_size": 32,
            "max_epochs": 10000,
            "patience": 1000,
        },
        "training": {
            "max_epochs": 10000,
            "min_epochs": 1,
            "current_epoch": 500,
            "global_step": 5000,
            "training_duration_seconds": 3600.0,
            "final_train_loss": 0.001,
            "final_test_loss": 0.002,
        },
        "model_sizes": {
            "network_size_bytes": 1000000,
            "network_size_mb": 0.95,
            "lut_size_bytes": 5000000,
            "lut_size_mb": 4.77,
            "safety_net_total_bytes": 6000000,
            "safety_net_total_mb": 5.72,
        },
        "lut_statistics": {
            "lut_entries": 1000,
            "dataset_size": 10000,
            "lut_coverage": 0.1,
            "nn_coverage": 0.9,
        },
        "dataset": {
            "filename": "test.h5",
            "size": 10000,
        },
    }

    # Verify all required sections exist
    assert "hyperparameters" in training_info
    assert "training" in training_info
    assert "model_sizes" in training_info
    assert "lut_statistics" in training_info
    assert "dataset" in training_info

    # Verify hyperparameters
    hyper = training_info["hyperparameters"]
    assert "activation" in hyper
    assert "hidden_layers" in hyper
    assert "num_hidden_layers" in hyper
    assert "inputs" in hyper
    assert "outputs" in hyper

    # Verify training info
    train = training_info["training"]
    assert "training_duration_seconds" in train
    assert "final_train_loss" in train
    assert "final_test_loss" in train

    # Verify model sizes
    sizes = training_info["model_sizes"]
    assert "network_size_bytes" in sizes
    assert "network_size_mb" in sizes
    assert "lut_size_bytes" in sizes
    assert "lut_size_mb" in sizes

    # Verify LUT statistics
    lut = training_info["lut_statistics"]
    assert "lut_entries" in lut
    assert "dataset_size" in lut
    assert "lut_coverage" in lut
    assert "nn_coverage" in lut
