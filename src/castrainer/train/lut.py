# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""K-d-tree backed lookup table utilities for SafetyNet."""

from __future__ import annotations

import sys
from gc import get_referents
from types import FunctionType, ModuleType
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from scipy import spatial as scipy_spatial

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Any

# cKDTree is a real, stable public attribute of scipy.spatial at
# runtime; ty's scipy stubs don't currently expose it via this
# compiled submodule.
SciPyKDTree = scipy_spatial.cKDTree  # ty: ignore[unresolved-attribute]

BLACKLIST = type, ModuleType, FunctionType


def get_object_size(obj: object) -> int:
    """Get the true size of an object in bytes.

    This counts Python object overhead and, for NumPy arrays and
    Torch tensors, their underlying data/storage payloads.

    Args:
        obj (object): Object to measure.

    Returns:
        int: Size of the object in bytes.

    Raises:
        TypeError: If `obj`'s type is in `BLACKLIST`.
    """
    if isinstance(obj, BLACKLIST):
        raise TypeError("getsize() does not take argument of type: " + str(type(obj)))

    seen_ids: set[int] = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj_ in objects:
            if isinstance(obj_, BLACKLIST):
                continue
            oid = id(obj_)
            if oid in seen_ids:
                continue
            seen_ids.add(oid)

            if isinstance(obj_, np.ndarray):
                size += int(obj_.nbytes) + sys.getsizeof(obj_)
                continue

            if isinstance(obj_, torch.Tensor):
                size += sys.getsizeof(obj_) + _tensor_storage_nbytes(obj_)
                continue

            size += sys.getsizeof(obj_)
            need_referents.append(obj_)

        objects = get_referents(*need_referents)

    return int(size)


def _tensor_to_array(values: torch.Tensor | Iterable[float]) -> np.ndarray:
    """Convert tensor-like values to a 1D float32 NumPy array.

    Args:
        values (torch.Tensor | Iterable[float]): The values to
            convert.

    Returns:
        np.ndarray: The values as a 1D float32 NumPy array.
    """
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().reshape(-1).to(dtype=torch.float32).numpy()
    return np.asarray(list(values), dtype=np.float32).reshape(-1)


def _to_float(value: object) -> float:
    """Convert a serialized value to float.

    Args:
        value (object): The value to convert.

    Returns:
        float: The value converted to float.
    """
    return float(value)  # ty: ignore[invalid-argument-type]


def _is_ckdtree_node(obj: object) -> bool:
    """Check if the object looks like a SciPy cKDTree node wrapper.

    Args:
        obj (object): The object to check.

    Returns:
        bool: True if `obj` exposes the expected node attributes.
    """
    return all(
        hasattr(obj, attr)
        for attr in ("data_points", "indices", "lesser", "greater", "split_dim")
    )


def _is_ckdtree(obj: object) -> bool:
    """Check if the object looks like a SciPy cKDTree wrapper.

    Args:
        obj (object): The object to check.

    Returns:
        bool: True if `obj` exposes the expected cKDTree attributes.
    """
    return all(hasattr(obj, attr) for attr in ("data", "indices", "tree", "query"))


def _tensor_storage_nbytes(tensor: torch.Tensor) -> int:
    """Return the size of a tensor's backing storage in bytes.

    Args:
        tensor (torch.Tensor): Tensor to measure.

    Returns:
        int: Size of the tensor's underlying storage in bytes.
    """
    try:
        storage = tensor.untyped_storage()
        return int(storage.nbytes())
    except Exception:  # ruff: ignore[blind-except]
        # torch's storage API has changed across versions; fall back
        # to the legacy accessor rather than assuming which is present.
        storage = tensor.storage()
        if hasattr(storage, "nbytes"):
            return int(storage.nbytes)
        return int(storage.size() * storage.element_size())


_LUT_ARRAY_NDIM = 2


class KDTreeLUT:
    """A k-d-tree backed LUT for exact SafetyNet lookups."""

    def __init__(self, points: np.ndarray, values: np.ndarray) -> None:
        """Build the LUT's k-d tree from parallel point/value arrays.

        Args:
            points (np.ndarray): 2D array of input points, one row
                per entry.
            values (np.ndarray): 2D array of output values, one row
                per entry.

        Raises:
            ValueError: If `points`/`values` aren't matching 2D
                arrays.
        """
        if points.ndim != _LUT_ARRAY_NDIM:
            raise ValueError("points must be a 2D array")
        if values.ndim != _LUT_ARRAY_NDIM:
            raise ValueError("values must be a 2D array")
        if len(points) != len(values):
            raise ValueError("points and values must have the same length")

        # cKDTree internally stores data in a contiguous C-backed
        # representation. Keep a reference to the tree's data array so
        # the LUT exposes the actual in-memory point buffer rather
        # than a separate copy.
        points_array = np.asarray(points, dtype=np.float64, order="C")
        self._values = np.asarray(values, dtype=np.float32, order="C")

        if len(points_array) == 0:
            self._tree: Any | None = None
            self._points = points_array
            self._dimensions = 0
        else:
            self._tree = SciPyKDTree(
                points_array,
                compact_nodes=True,
                balanced_tree=True,
            )
            self._points = self._tree.data
            self._dimensions = int(self._points.shape[1])

    @classmethod
    def empty(
        cls,
        input_dim: int = 0,
        output_dim: int = 0,
    ) -> KDTreeLUT:
        """Create an empty LUT with optional dimensions.

        Args:
            input_dim (int): Number of input dimensions.
            output_dim (int): Number of output dimensions.

        Returns:
            KDTreeLUT: An empty LUT.
        """
        points = np.empty((0, input_dim), dtype=np.float32)
        values = np.empty((0, output_dim), dtype=np.float32)
        return cls(points, values)

    @classmethod
    def from_items(
        cls,
        items: Iterable[tuple[torch.Tensor, torch.Tensor]],
    ) -> KDTreeLUT:
        """Build a LUT from tensor pairs.

        Args:
            items (Iterable[tuple[torch.Tensor, torch.Tensor]]): The
                (input, output) tensor pairs to build the LUT from.

        Returns:
            KDTreeLUT: The resulting LUT.
        """
        deduplicated: dict[tuple[float, ...], np.ndarray] = {}
        input_dim = 0
        output_dim = 0

        for key, value in items:
            key_array = _tensor_to_array(key)
            value_array = _tensor_to_array(value)
            input_dim = key_array.size
            output_dim = value_array.size
            deduplicated[tuple(key_array.tolist())] = value_array

        if not deduplicated:
            return cls.empty(input_dim=input_dim, output_dim=output_dim)

        points = np.asarray(list(deduplicated.keys()), dtype=np.float32)
        values = np.asarray(list(deduplicated.values()), dtype=np.float32)
        return cls(points, values)

    @classmethod
    def from_serialized_entries(
        cls,
        entries: Iterable[dict[str, object]],
    ) -> KDTreeLUT:
        """Build a LUT from serialized JSON entries.

        Args:
            entries (Iterable[dict[str, object]]): The serialized LUT
                entries to build from.

        Returns:
            KDTreeLUT: The resulting LUT.
        """
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for entry in entries:
            raw_inputs = entry.get("inputs", {})
            outputs_list = entry.get("outputs", [])
            if not isinstance(raw_inputs, dict):
                continue
            if not isinstance(outputs_list, list):
                continue
            inputs_dict = cast("dict[str, object]", raw_inputs)

            sorted_keys = sorted(inputs_dict.keys())
            x_values = [
                _to_float(inputs_dict[k]) for k in sorted_keys if k in inputs_dict
            ]
            y_values = [_to_float(v) for v in outputs_list]
            pairs.append((
                torch.tensor(x_values, dtype=torch.float32),
                torch.tensor(y_values, dtype=torch.float32),
            ))

        return cls.from_items(pairs)

    def __len__(self) -> int:
        if self._tree is None:
            return 0
        return int(self._tree.n)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, torch.Tensor):
            try:
                self[key]
            except KeyError:
                return False
            return True
        return False

    def __getitem__(self, key: torch.Tensor | Iterable[float]) -> torch.Tensor:
        if len(self) == 0:
            raise KeyError("LUT is empty")

        query = _tensor_to_array(key).astype(np.float64, copy=False)
        if self._dimensions and query.size != self._dimensions:
            raise KeyError("LUT key has an unexpected dimension")

        if self._tree is None:
            raise KeyError("LUT is empty")

        distance, index = self._tree.query(query, k=1)
        # Exact-match lookup by design: `query` is derived from the
        # same float64 values used to build the tree, so a real hit
        # reproduces the stored point bit-for-bit (distance == 0.0).
        # A tolerance-based comparison would wrongly accept near misses.
        if not np.isfinite(distance) or distance != 0.0:  # ruff: ignore[float-equality-comparison]
            raise KeyError("LUT entry not found")

        return torch.tensor(self._values[int(index)], dtype=torch.float32)

    def items(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over LUT entries as tensor pairs.

        Yields:
            tuple[torch.Tensor, torch.Tensor]: Each stored (input,
                output) tensor pair.
        """
        for point, value in zip(self._points, self._values, strict=False):
            yield (
                torch.tensor(point, dtype=torch.float32),
                torch.tensor(value, dtype=torch.float32),
            )

    @property
    def points(self) -> np.ndarray:
        """The stored input points."""
        return self._points

    @property
    def values(self) -> np.ndarray:
        """The stored outputs."""
        return self._values

    def to_serializable_entries(self) -> list[dict[str, object]]:
        """Convert LUT entries to the JSON schema representation.

        Returns:
            list[dict[str, object]]: The LUT's entries as
                JSON-serializable dictionaries.
        """
        data: list[dict[str, object]] = []
        for x, y in self.items():
            inputs_dict = {
                f"input_{i}": float(v.item()) for i, v in enumerate(x.flatten())
            }
            outputs_list = [float(v.item()) for v in y.flatten()]
            data.append({"inputs": inputs_dict, "outputs": outputs_list})
        return data

    def size_bytes(self) -> int:
        """Measure the in-memory size of the SciPy cKDTree-backed LUT.

        This counts:
        - the Python wrapper object
        - the tree's C-backed arrays (`data`, `indices`, `maxes`,
            `mins`)
        - the node wrapper graph reachable from `tree.tree`
        - the stored values array
        - any nested NumPy arrays / Torch tensors without
            double-counting

        Returns:
            int: The estimated total size, in bytes.
        """
        seen_ids: set[int] = set()

        def add(obj: object | None) -> int:
            if obj is None or id(obj) in seen_ids:
                return 0
            oid = id(obj)
            seen_ids.add(oid)

            total = sys.getsizeof(obj)

            if isinstance(obj, np.ndarray):
                return total + int(obj.nbytes)

            if isinstance(obj, torch.Tensor):
                return total + _tensor_storage_nbytes(obj)

            if _is_ckdtree(obj):
                total += add(getattr(obj, "data", None))
                total += add(getattr(obj, "indices", None))
                total += add(getattr(obj, "maxes", None))
                total += add(getattr(obj, "mins", None))
                total += add(getattr(obj, "tree", None))
                return total

            if _is_ckdtree_node(obj):
                total += add(getattr(obj, "data_points", None))
                total += add(getattr(obj, "indices", None))
                total += add(getattr(obj, "lesser", None))
                total += add(getattr(obj, "greater", None))
                return total

            return total

        total = sys.getsizeof(self)
        total += add(self._points)
        total += add(self._values)
        total += add(self._tree)
        return int(total)
