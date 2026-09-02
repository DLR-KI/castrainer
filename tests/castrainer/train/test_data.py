# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for CASDataset and SharedCASDataset."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from castrainer.train.data import CASDataset, SharedCASDataset


def _write_h5(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    with h5py.File(str(path), "w") as f:
        f.create_dataset("X", data=x)
        f.create_dataset("y", data=y)


def _sample_h5(tmp_path: Path) -> Path:
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    y = np.array([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1], [0.0, 0.0, 1.0]], dtype=np.float32)
    path = tmp_path / "data.h5"
    _write_h5(path, x, y)
    return path


def test_cas_dataset_loads_data(tmp_path: Path) -> None:
    path = _sample_h5(tmp_path)
    dataset = CASDataset(path)

    assert len(dataset) == 3
    assert dataset.x.shape == (3, 2)
    assert dataset.y.shape == (3, 3)
    assert dataset.one_hot is False


def test_cas_dataset_getitem(tmp_path: Path) -> None:
    path = _sample_h5(tmp_path)
    dataset = CASDataset(path)

    x0, y0 = dataset[0]
    assert torch.equal(x0, torch.tensor([1.0, 2.0]))
    assert torch.equal(y0, torch.tensor([0.1, 0.9, 0.0]))


def test_cas_dataset_one_hot_encodes_argmax(tmp_path: Path) -> None:
    path = _sample_h5(tmp_path)
    dataset = CASDataset(path, one_hot=True)

    assert dataset.one_hot is True
    expected = torch.tensor([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert torch.equal(dataset.y, expected)


def test_shared_cas_dataset_create_and_attach(tmp_path: Path) -> None:
    path = _sample_h5(tmp_path)
    base = CASDataset(path)

    shared = SharedCASDataset(base, create=True)
    try:
        assert len(shared) == len(base)
        assert torch.equal(shared.x, base.x)
        assert torch.equal(shared.y, base.y)
        x0, y0 = shared[0]
        assert torch.equal(x0, base.x[0])
        assert torch.equal(y0, base.y[0])

        attached = SharedCASDataset(shared, create=False)
        try:
            assert torch.equal(attached.x, base.x)
            assert torch.equal(attached.y, base.y)
        finally:
            attached.cleanup()
    finally:
        shared.cleanup()


def test_shared_cas_dataset_create_requires_cas_dataset(tmp_path: Path) -> None:
    path = _sample_h5(tmp_path)
    base = CASDataset(path)
    shared = SharedCASDataset(base, create=True)
    try:
        with pytest.raises(ValueError, match="Creating shared memory"):
            SharedCASDataset(shared, create=True)
    finally:
        shared.cleanup()


def test_shared_cas_dataset_attach_requires_shared_dataset(tmp_path: Path) -> None:
    path = _sample_h5(tmp_path)
    base = CASDataset(path)
    with pytest.raises(ValueError, match="Attaching shared memory"):
        SharedCASDataset(base, create=False)


def test_shared_cas_dataset_cleanup_idempotent(tmp_path: Path) -> None:
    path = _sample_h5(tmp_path)
    base = CASDataset(path)
    shared = SharedCASDataset(base, create=True)
    shared.cleanup()
    # Calling cleanup twice must not raise even though the shared
    # memory segments are already closed/unlinked.
    shared.cleanup()
