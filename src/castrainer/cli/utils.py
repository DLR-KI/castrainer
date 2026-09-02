# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Shared utilities for CLI commands."""

import os

import torch
from loguru import logger


def set_thread_limits(num_threads: int | None) -> None:
    """Set thread limits for CPU operations.

    Args:
        num_threads (int | None): Number of threads to use. If None,
            no limits are set.
    """
    if num_threads is None:
        return

    # Set environment variables
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)

    # Set PyTorch thread limit
    torch.set_num_threads(num_threads)

    logger.info(f"Set CPU thread limit to {num_threads}")
