# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for the KDTreeLUT k-d-tree backed lookup table."""

import numpy as np
import pytest
import torch

from castrainer.train.lut import KDTreeLUT, get_object_size


def _sample_lut() -> KDTreeLUT:
    return KDTreeLUT.from_items([
        (torch.tensor([1.0, 2.0]), torch.tensor([0.5, 0.25])),
        (torch.tensor([3.0, 4.0]), torch.tensor([0.7, 0.35])),
    ])


def test_kdtree_lut_rejects_1d_points() -> None:
    with pytest.raises(ValueError, match="points must be a 2D array"):
        KDTreeLUT(np.array([1.0, 2.0]), np.array([[1.0]]))


def test_kdtree_lut_rejects_1d_values() -> None:
    with pytest.raises(ValueError, match="values must be a 2D array"):
        KDTreeLUT(np.array([[1.0, 2.0]]), np.array([1.0]))


def test_kdtree_lut_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        KDTreeLUT(np.array([[1.0], [2.0]]), np.array([[1.0]]))


def test_kdtree_lut_empty_classmethod() -> None:
    lut = KDTreeLUT.empty(input_dim=2, output_dim=1)
    assert len(lut) == 0
    assert lut.points.shape == (0, 2)
    assert lut.values.shape == (0, 1)


def test_kdtree_lut_from_items_deduplicates() -> None:
    lut = KDTreeLUT.from_items([
        (torch.tensor([1.0, 2.0]), torch.tensor([0.1])),
        (torch.tensor([1.0, 2.0]), torch.tensor([0.9])),
    ])
    assert len(lut) == 1
    # Later duplicate keys overwrite earlier ones.
    assert torch.equal(lut[torch.tensor([1.0, 2.0])], torch.tensor([0.9]))


def test_kdtree_lut_from_items_empty() -> None:
    lut = KDTreeLUT.from_items([])
    assert len(lut) == 0


def test_kdtree_lut_getitem_hit() -> None:
    lut = _sample_lut()
    result = lut[torch.tensor([3.0, 4.0])]
    assert torch.equal(result, torch.tensor([0.7, 0.35]))


def test_kdtree_lut_getitem_miss_raises() -> None:
    lut = _sample_lut()
    with pytest.raises(KeyError, match="not found"):
        lut[torch.tensor([9.0, 9.0])]


def test_kdtree_lut_getitem_wrong_dimension_raises() -> None:
    lut = _sample_lut()
    with pytest.raises(KeyError, match="unexpected dimension"):
        lut[torch.tensor([1.0, 2.0, 3.0])]


def test_kdtree_lut_getitem_empty_raises() -> None:
    lut = KDTreeLUT.empty(input_dim=2, output_dim=1)
    with pytest.raises(KeyError, match="empty"):
        lut[torch.tensor([1.0, 2.0])]


def test_kdtree_lut_contains() -> None:
    lut = _sample_lut()
    assert torch.tensor([1.0, 2.0]) in lut
    assert torch.tensor([9.0, 9.0]) not in lut
    assert "not-a-tensor" not in lut


def test_kdtree_lut_items_round_trip() -> None:
    lut = _sample_lut()
    pairs = list(lut.items())
    assert len(pairs) == 2
    for key, _value in pairs:
        assert key in lut


def test_kdtree_lut_to_serializable_entries() -> None:
    lut = _sample_lut()
    entries = lut.to_serializable_entries()
    assert len(entries) == 2
    for entry in entries:
        assert set(entry.keys()) == {"inputs", "outputs"}
        outputs = entry["outputs"]
        assert isinstance(outputs, list)
        assert len(outputs) == 2


def test_kdtree_lut_from_serialized_entries_round_trip() -> None:
    original = _sample_lut()
    entries = original.to_serializable_entries()

    rebuilt = KDTreeLUT.from_serialized_entries(entries)

    assert len(rebuilt) == len(original)
    assert torch.equal(rebuilt[torch.tensor([1.0, 2.0])], torch.tensor([0.5, 0.25]))


def test_kdtree_lut_from_serialized_entries_skips_malformed() -> None:
    entries: list[dict[str, object]] = [
        {"inputs": {"a": 1.0}, "outputs": [0.5]},
        {"inputs": "not-a-dict", "outputs": [0.5]},
        {"inputs": {"a": 1.0}, "outputs": "not-a-list"},
    ]
    lut = KDTreeLUT.from_serialized_entries(entries)
    assert len(lut) == 1


def test_kdtree_lut_size_bytes_positive() -> None:
    lut = _sample_lut()
    assert lut.size_bytes() > 0


def test_kdtree_lut_size_bytes_empty() -> None:
    lut = KDTreeLUT.empty(input_dim=2, output_dim=1)
    assert lut.size_bytes() > 0


def test_get_object_size_simple_values() -> None:
    assert get_object_size(42) > 0
    assert get_object_size([1, 2, 3]) > 0


def test_get_object_size_numpy_array() -> None:
    arr = np.zeros((10, 10), dtype=np.float32)
    assert get_object_size(arr) >= arr.nbytes


def test_get_object_size_torch_tensor() -> None:
    tensor = torch.zeros(10, 10)
    assert get_object_size(tensor) > 0


def test_get_object_size_rejects_type_objects() -> None:
    with pytest.raises(TypeError):
        get_object_size(int)
