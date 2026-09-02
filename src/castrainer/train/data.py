# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""The dataset definition for the training data."""

import atexit
import multiprocessing.shared_memory as sm
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset


class CASDataset(Dataset):
    """The dataset class for the training data.

    Args:
        file (Path): The path to the HDF5 file containing the training
            data.
        one_hot (bool): Whether to use one-hot encoding for the target.
    """

    def __init__(
        self,
        file: Path,
        one_hot: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]
    ) -> None:
        """Load training data from an HDF5 file into tensors.

        Args:
            file (Path): The path to the HDF5 file containing the
                training data.
            one_hot (bool): Whether to use one-hot encoding for the
                target.
        """
        self.file = file
        self.one_hot = one_hot

        with h5py.File(name=str(self.file.resolve()), mode="r") as f:
            x: npt.NDArray = cast("h5py.Dataset", f["X"])[:]
            y: npt.NDArray = cast("h5py.Dataset", f["y"])[:]
            if one_hot:
                y_one_hot = np.zeros_like(y)
                max_indices = np.argmax(y, axis=1)
                y_one_hot[np.arange(y.shape[0]), max_indices] = 1
                y = y_one_hot

        self.x = torch.tensor(x, dtype=torch.float32).detach()
        self.y = torch.tensor(y, dtype=torch.float32).detach()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class SharedCASDataset(Dataset):
    """Dataset class for training data using shared memory.

    Args:
        dataset ("CASDataset | SharedCASDataset"): An existing
            CASDataset or SharedCASDataset instance.
        create (bool): If True, creates new shared memory (only in the
            main process).

    Raises:
        ValueError: If `dataset` is not compatible.
    """

    def __init__(
        self,
        dataset: "CASDataset | SharedCASDataset",
        create: bool = False,  # ruff: ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]
    ) -> None:
        """Create or attach to shared memory for an existing dataset.

        Args:
            dataset (CASDataset | SharedCASDataset): An existing
                CASDataset or SharedCASDataset instance.
            create (bool): If True, creates new shared memory (only in
                the main process).

        Raises:
            ValueError: If `dataset` is incompatible with `create`.
        """
        self.file: Path = dataset.file
        self.one_hot: bool = dataset.one_hot

        self.created = create  # Track if this process created shared memory

        if create:
            if not isinstance(dataset, CASDataset):
                raise ValueError("Creating shared memory requires a CASDataset.")
            x = dataset.x.numpy()
            y = dataset.y.numpy()

            # Store shape and dtype
            self.shape_x, self.shape_y = x.shape, y.shape
            self.dtype_x, self.dtype_y = x.dtype, y.dtype

            # Create shared memory buffers
            self.shm_x = sm.SharedMemory(create=True, size=x.nbytes)
            self.shm_y = sm.SharedMemory(create=True, size=y.nbytes)

            # Copy data into shared memory
            np.copyto(
                np.ndarray(self.shape_x, dtype=self.dtype_x, buffer=self.shm_x.buf), x
            )
            np.copyto(
                np.ndarray(self.shape_y, dtype=self.dtype_y, buffer=self.shm_y.buf), y
            )

        else:
            if isinstance(dataset, CASDataset):
                raise ValueError("Attaching shared memory requires a SharedCASDataset.")
            # Attach to existing shared memory
            self.shm_x = sm.SharedMemory(name=dataset.shm_x.name)
            self.shm_y = sm.SharedMemory(name=dataset.shm_y.name)

            self.shape_x, self.shape_y = dataset.shape_x, dataset.shape_y
            self.dtype_x, self.dtype_y = dataset.dtype_x, dataset.dtype_y

        # Map shared memory to NumPy arrays and convert to torch tensors
        self.dataset_x_np: npt.NDArray = np.ndarray(
            self.shape_x, dtype=self.dtype_x, buffer=self.shm_x.buf
        )
        self.dataset_y_np: npt.NDArray = np.ndarray(
            self.shape_y, dtype=self.dtype_y, buffer=self.shm_y.buf
        )

        self.x = torch.from_numpy(self.dataset_x_np).detach()
        self.y = torch.from_numpy(self.dataset_y_np).detach()

        # Register cleanup function
        atexit.register(self.cleanup)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]

    def cleanup(self) -> None:
        """Properly closes and unlinks shared memory."""
        try:
            self.shm_x.close()
            self.shm_y.close()
            if self.created:  # Only unlink if this process created it
                self.shm_x.unlink()
                self.shm_y.unlink()
        except FileNotFoundError:
            pass  # Ignore if already unlinked
