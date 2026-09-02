# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for SafetyNet directory loading helpers."""

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from castrainer.safetynet.io import (
    LoadedSafetyNet,
    _infer_model_architecture,
    load_safetynet_directory,
)
from castrainer.train.lut import KDTreeLUT
from castrainer.train.net import Net


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)


def _build_system(
    safetynet_dir: Path,
    sys_name: str,
    *,
    with_lut: bool = True,
) -> None:
    sys_dir = safetynet_dir / sys_name
    sys_dir.mkdir(parents=True)

    model = Net(inputs=3, outputs=2, hidden_layers=[4], activation=nn.ReLU)
    torch.save(model.state_dict(), sys_dir / "net.pt")

    networks = [{"file": "net.pt", "networkFormat": "torch", "if": {}}]
    luts = []
    if with_lut:
        lut = KDTreeLUT.from_items([
            (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.1, 0.9])),
        ])
        _write_json(
            sys_dir / "net_lut.json",
            {
                "version": "1.0.0",
                "datatype": "float32",
                "numberOutputs": 2,
                "data": lut.to_serializable_entries(),
            },
        )
        luts = [{"file": "net_lut.json", "lutFormat": "snet", "if": {}}]

    manifest = {
        "version": "1.0.0",
        "datatype": "float32",
        "numberOutputs": 2,
        "inputs": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
        "networks": networks,
        "luts": luts,
    }
    _write_json(sys_dir / f"{sys_name}.json", manifest)


def test_load_safetynet_directory_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Directory not found"):
        load_safetynet_directory(tmp_path / "nonexistent")


def test_load_safetynet_directory_no_valid_systems(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No valid system manifests"):
        load_safetynet_directory(tmp_path)


def test_load_safetynet_directory_unknown_system(tmp_path: Path) -> None:
    _build_system(tmp_path, "vcas")
    with pytest.raises(ValueError, match="System 'hcas' not found"):
        load_safetynet_directory(tmp_path, system="hcas")


def test_load_safetynet_directory_all_systems(tmp_path: Path) -> None:
    _build_system(tmp_path, "vcas")
    _build_system(tmp_path, "hcas")

    loaded = load_safetynet_directory(tmp_path)

    assert set(loaded.keys()) == {"vcas", "hcas"}
    for bundles in loaded.values():
        assert len(bundles) == 1
        bundle = bundles[0]
        assert isinstance(bundle, LoadedSafetyNet)
        assert bundle.model.inputs == 3
        assert bundle.model.outputs == 2
        assert bundle.lut_file is not None
        assert len(bundle.safetynet.lut) == 1


def test_load_safetynet_directory_single_system(tmp_path: Path) -> None:
    _build_system(tmp_path, "vcas")
    _build_system(tmp_path, "hcas")

    loaded = load_safetynet_directory(tmp_path, system="vcas")

    assert set(loaded.keys()) == {"vcas"}


def test_load_safetynet_directory_missing_network_file_skipped(
    tmp_path: Path,
) -> None:
    _build_system(tmp_path, "vcas")
    manifest_path = tmp_path / "vcas" / "vcas.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["networks"].append({"file": "missing.pt", "if": {}})
    _write_json(manifest_path, manifest)

    loaded = load_safetynet_directory(tmp_path, system="vcas")

    # Only the network with a real file on disk is loaded.
    assert len(loaded["vcas"]) == 1


def test_load_safetynet_directory_missing_lut_file(tmp_path: Path) -> None:
    _build_system(tmp_path, "vcas", with_lut=False)
    manifest_path = tmp_path / "vcas" / "vcas.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["luts"] = [{"file": "missing_lut.json", "if": {}}]
    _write_json(manifest_path, manifest)

    loaded = load_safetynet_directory(tmp_path, system="vcas")

    bundle = loaded["vcas"][0]
    assert bundle.lut_file is not None
    assert bundle.lut_serialized_size_bytes is None
    # The SafetyNet keeps its default (empty) LUT.
    assert len(bundle.safetynet.lut) == 0


def test_infer_model_architecture_from_real_state_dict() -> None:
    model = Net(inputs=6, outputs=9, hidden_layers=[10, 10], activation=nn.ReLU)
    state = model.state_dict()

    inputs, outputs, hidden_layers = _infer_model_architecture(state, 0, 0)

    assert inputs == 6
    assert outputs == 9
    assert hidden_layers == [10, 10]


def test_infer_model_architecture_falls_back_when_no_encoder_keys() -> None:
    inputs, outputs, hidden_layers = _infer_model_architecture({}, 4, 5)
    assert (inputs, outputs, hidden_layers) == (4, 5, [])


def test_infer_model_architecture_skips_malformed_keys() -> None:
    state = {"encoder.notanumber.weight": torch.zeros(2, 2)}
    inputs, outputs, hidden_layers = _infer_model_architecture(state, 4, 5)
    assert (inputs, outputs, hidden_layers) == (4, 5, [])
