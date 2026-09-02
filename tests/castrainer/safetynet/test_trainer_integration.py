# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""End-to-end tests for SafetyNetTrainer.train_system with tiny data."""

import json
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from torch import nn

from castrainer.safetynet.config import HCAS_CONFIG
from castrainer.safetynet.trainer import SafetyNetTrainer, TrainingConfig


def _write_h5(path: Path) -> None:
    rng = np.random.default_rng(0)
    with h5py.File(str(path), "w") as f:
        f.create_dataset("X", data=rng.random((8, 4), dtype=np.float32))
        f.create_dataset("y", data=rng.random((8, 9), dtype=np.float32))


def test_train_system_vcas_end_to_end(tmp_path: Path) -> None:
    """Real (tiny) training for all 9 VCAS subsystems produces a manifest."""
    h5_path = tmp_path / "data.h5"
    _write_h5(h5_path)

    output_dir = tmp_path / "safetynet"
    config = TrainingConfig(
        hidden_nodes=[4],
        hidden_layers=[1],
        batch_size=4,
        max_epochs=1,
        patience=1,
        nproc=1,
        enable_progress_bar=False,
    )
    trainer = SafetyNetTrainer(config=config, output_dir=output_dir)

    with patch.object(SafetyNetTrainer, "_get_data_path", return_value=h5_path):
        trainer.train_system("vcas", activation_fn=nn.ReLU)

    vcas_dir = output_dir / "vcas"
    manifest_path = vcas_dir / "vcas.json"
    assert manifest_path.exists()

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert len(manifest["networks"]) == 9
    assert len(manifest["luts"]) == 9

    # Each subsystem produced a model, LUT, and training-info file.
    assert (vcas_dir / "vcas_01.pt").exists()
    assert (vcas_dir / "vcas_01_lut.json").exists()
    assert (vcas_dir / "vcas_01_training_info.json").exists()


def test_train_system_hcas_reduced_subsystems(tmp_path: Path) -> None:
    """Real (tiny) training for a reduced HCAS subsystem count."""
    h5_path = tmp_path / "data.h5"
    _write_h5(h5_path)

    output_dir = tmp_path / "safetynet"
    config = TrainingConfig(
        hidden_nodes=[4],
        hidden_layers=[1],
        batch_size=4,
        max_epochs=1,
        patience=1,
        nproc=1,
        enable_progress_bar=False,
    )
    trainer = SafetyNetTrainer(config=config, output_dir=output_dir)

    # HCAS normally has 40 subsystems; shrink it so the real training
    # loop stays fast while still exercising the HCAS-specific naming
    # branch (pra/tau).
    with (
        patch.object(SafetyNetTrainer, "_get_data_path", return_value=h5_path),
        patch.dict(HCAS_CONFIG, {"num_subsystems": 1}),
    ):
        trainer.train_system("hcas", activation_fn=nn.ReLU)

    hcas_dir = output_dir / "hcas"
    manifest_path = hcas_dir / "hcas.json"
    assert manifest_path.exists()

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert len(manifest["networks"]) == 1
    assert manifest["networks"][0]["if"] == {
        "pra": {"minimum": 0.0, "maximum": 0.0},
        "tau": {"minimum": 0.0, "maximum": 0.0},
    }
    assert (hcas_dir / "hcas_pra0_tau00.pt").exists()


def test_train_system_skips_missing_data_file(tmp_path: Path) -> None:
    """Subsystems whose data file is missing are skipped, not fatal."""
    output_dir = tmp_path / "safetynet"
    config = TrainingConfig(nproc=1, enable_progress_bar=False)
    trainer = SafetyNetTrainer(config=config, output_dir=output_dir)

    with (
        patch.object(
            SafetyNetTrainer, "_get_data_path", return_value=tmp_path / "missing.h5"
        ),
        patch.dict(HCAS_CONFIG, {"num_subsystems": 1}),
    ):
        trainer.train_system("hcas", activation_fn=nn.ReLU)

    manifest_path = output_dir / "hcas" / "hcas.json"
    assert manifest_path.exists()
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["networks"] == []
    assert manifest["luts"] == []


def test_train_system_unknown_system_raises(tmp_path: Path) -> None:
    trainer = SafetyNetTrainer(output_dir=tmp_path)
    with pytest.raises(ValueError, match="Unknown system"):
        trainer.train_system("unknown", activation_fn=nn.ReLU)
