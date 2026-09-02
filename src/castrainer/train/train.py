# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Train models using PyTorch Lightning."""

import datetime
import json
import multiprocessing.managers
import operator
import os
import sys
import time
from collections.abc import Iterable
from datetime import timedelta
from gc import get_referents
from pathlib import Path
from types import FunctionType, ModuleType
from typing import TYPE_CHECKING, Literal, TypedDict

import lightning as L  # ruff: ignore[lowercase-imported-as-non-lowercase]
import orjson
import pynvml
import torch
import torch.multiprocessing as mp
from lightning.fabric.plugins.precision.precision import (
    _PRECISION_INPUT,
)
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.plugins import _PLUGIN_INPUT
from lightning.pytorch.profilers import Profiler
from lightning.pytorch.strategies import Strategy
from lightning.pytorch.trainer.connectors.accelerator_connector import (
    _LITERAL_WARN,
)
from loguru import logger
from torch.utils.data import DataLoader

from castrainer.train.data import CASDataset, SharedCASDataset
from castrainer.train.net import Net
from castrainer.train.safetynet import SafetyNet

if TYPE_CHECKING:
    from multiprocessing.context import SpawnProcess

torch.multiprocessing.set_start_method("spawn", force=True)

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType

TIMESTAMP = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")


class LTrainerKWArgs(TypedDict, total=False):
    """Keyword arguments for the PyTorch Lightning Trainer.

    Copied from the PyTorch Lightning documentation.
    """

    accelerator: str | Accelerator | None
    strategy: str | Strategy | None
    devices: list[int] | str | int | None
    num_nodes: int | None
    precision: _PRECISION_INPUT | None
    logger: Logger | Iterable[Logger] | bool | None
    callbacks: list[Callback] | Callback | None
    fast_dev_run: int | bool | None
    max_epochs: int | None
    min_epochs: int | None
    max_steps: int | None
    min_steps: int | None
    max_time: str | timedelta | dict[str, int] | None
    limit_train_batches: int | float | None
    limit_val_batches: int | float | None
    limit_test_batches: int | float | None
    limit_predict_batches: int | float | None
    overfit_batches: int | float | None
    val_check_interval: int | float | None
    check_val_every_n_epoch: int | None
    num_sanity_val_steps: int | None
    log_every_n_steps: int | None
    enable_checkpointing: bool | None
    enable_progress_bar: bool | None
    enable_model_summary: bool | None
    accumulate_grad_batches: int | None
    gradient_clip_val: int | float | None
    gradient_clip_algorithm: str | None
    deterministic: bool | _LITERAL_WARN | None
    benchmark: bool | None
    inference_mode: bool | None
    use_distributed_sampler: bool | None
    profiler: Profiler | str | None
    detect_anomaly: bool | None
    barebones: bool | None
    plugins: _PLUGIN_INPUT | list[_PLUGIN_INPUT] | None
    sync_batchnorm: bool | None
    reload_dataloaders_every_n_epochs: int | None
    default_root_dir: _PATH | None


def __getsize(obj: object) -> int:
    """Get the true size of an object in bytes.

    From: https://stackoverflow.com/a/30316760

    Args:
        obj (object): Object to get the size of.

    Returns:
        int: Size of the object in bytes.

    Raises:
        TypeError: If the object is in the blacklist.
    """
    if isinstance(obj, BLACKLIST):
        raise TypeError("getsize() does not take argument of type: " + str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj_ in objects:
            if not isinstance(obj_, BLACKLIST) and id(obj_) not in seen_ids:
                seen_ids.add(id(obj_))
                size += sys.getsizeof(obj_)
                need_referents.append(obj_)
        objects = get_referents(*need_referents)
    return size


def __train(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    model: Net,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    trainer_kwargs: LTrainerKWArgs,
    results: list[dict] | multiprocessing.managers.ListProxy,
    output_dir: Path | None = None,
) -> None:
    """Train the models and fill the results dict.

    Args:
        model (Net): List of models to train.
        train_dataloader (DataLoader): Data loader for the training
            data.
        test_dataloader (DataLoader): Data loader for the test data.
        trainer_kwargs (LTrainerKWArgs): Trainer keyword arguments.
        results (list[dict] | multiprocessing.managers.ListProxy):
            Dictionary to store the results.
        output_dir (Path | None): Optional directory to save training
            info.
    """
    start_time = time.perf_counter()

    trainer = L.Trainer(**trainer_kwargs)  # type: ignore
    trainer.fit(model=model, train_dataloaders=train_dataloader)

    result = trainer.test(model=model, dataloaders=test_dataloader)

    dataset: CASDataset = train_dataloader.dataset  # type: ignore

    snet = SafetyNet(model)
    snet.fill(dataset)

    end_time = time.perf_counter()
    training_duration = end_time - start_time

    result_data = {
        "results": result,
        "current_epoch": trainer.current_epoch,
        "max_epoch": trainer.max_epochs,
        "min_epoch": trainer.min_epochs,
        "global_step": trainer.global_step,
        "min_steps": trainer.min_steps,
        "max_steps": trainer.max_steps,
        "hidden_layers": model.hidden_layers,
        "inputs": len(dataset.x[0]),
        "outputs": len(dataset.y[0]),
        "activation": str(model.activation()),
        "dataset_len": len(dataset),
        "lut_len": len(snet.lut),
        "in_lut": len(snet.lut) / len(dataset),
        "in_nn": 1 - len(snet.lut) / len(dataset),
        "lut_size_bytes": __getsize(snet.lut),
        "net_size_bytes": __getsize(model),
        "safety_net_size_bytes": __getsize(snet),
        "one_hot": dataset.one_hot,
        "dataset_filename": str(dataset.file),
        "training_duration_seconds": training_duration,
    }

    results.append(result_data)

    if isinstance(results, multiprocessing.managers.ListProxy):
        results = results._getvalue()  # ruff: ignore[private-member-access]

    logger.info(orjson.dumps(results[-1]).decode("utf-8") + ",")

    with Path(
        Path(dataset.file).parent,
        Path(dataset.file).stem + f"_results-{TIMESTAMP}.json",
    ).open("w", encoding="utf-8") as f:
        json.dump(results, f)

    # Save training info file if output_dir is specified
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        info_file = output_dir / "training_info.json"

        # Extract final loss from training results
        final_train_loss = None
        final_test_loss = None
        if result:
            final_test_loss = result[0].get("test_loss")

        # Get training loss from trainer logs
        if (
            hasattr(trainer, "callback_metrics")
            and "train_loss" in trainer.callback_metrics
        ):
            final_train_loss = float(trainer.callback_metrics["train_loss"].item())

        training_info = {
            "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
            "hyperparameters": {
                "activation": str(model.activation()),
                "hidden_layers": model.hidden_layers,
                "num_hidden_layers": len(model.hidden_layers),
                "inputs": len(dataset.x[0]),
                "outputs": len(dataset.y[0]),
                "one_hot": dataset.one_hot,
            },
            "training": {
                "max_epochs": trainer.max_epochs,
                "min_epochs": trainer.min_epochs,
                "current_epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
                "training_duration_seconds": training_duration,
                "final_train_loss": final_train_loss,
                "final_test_loss": final_test_loss,
            },
            "model_sizes": {
                "network_size_bytes": __getsize(model),
                "network_size_mb": __getsize(model) / (1024 * 1024),
                "lut_size_bytes": __getsize(snet.lut),
                "lut_size_mb": __getsize(snet.lut) / (1024 * 1024),
                "safety_net_total_bytes": __getsize(snet),
                "safety_net_total_mb": __getsize(snet) / (1024 * 1024),
            },
            "lut_statistics": {
                "lut_entries": len(snet.lut),
                "dataset_size": len(dataset),
                "lut_coverage": len(snet.lut) / len(dataset),
                "nn_coverage": 1 - len(snet.lut) / len(dataset),
            },
            "dataset": {
                "filename": str(dataset.file),
                "size": len(dataset),
            },
        }

        with info_file.open("w", encoding="utf-8") as f:
            json.dump(training_info, f, indent=2)

        logger.info(f"Training info saved to {info_file}")


def __sequential(
    models: list[Net],
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    trainer_kwargs: LTrainerKWArgs,
    output_dir: Path | None = None,
) -> None:
    """Train models sequentially.

    Train the models sequentially and store the results in the results
    dictionary. This scheduler can run on a CPU-only machine.

    Args:
        models (list[Net]): List of models to train.
        train_dataloader (DataLoader): Data loader for the training
            data.
        test_dataloader (DataLoader): Data loader for the test data.
        trainer_kwargs (LTrainerKWArgs): Trainer keyword arguments.
        output_dir (Path | None): Optional directory to save training
            info.
    """
    results: list[dict] = []
    for model in models:
        __train(
            model,
            train_dataloader,
            test_dataloader,
            trainer_kwargs,
            results,
            output_dir,
        )
    logger.info(orjson.dumps(results).decode("utf-8"))


def _query_gpu_status(dev_idx: int) -> tuple[int, float, float] | None:
    """Query one GPU's free/total memory via pynvml.

    Args:
        dev_idx (int): GPU device index.

    Returns:
        tuple[int, float, float] | None: A (dev_idx, free_gb,
            total_gb) tuple, or None if the query failed.
    """
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(dev_idx)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    except pynvml.NVMLError as e:
        logger.error(f"Error accessing GPU {dev_idx}: {e}")
        return None

    free_gb = info.free / 1024**3
    total_gb = info.total / 1024**3
    used_fraction = 1.0 - (info.free / info.total)
    logger.debug(
        f"GPU {dev_idx}: {free_gb:.2f}/{total_gb:.2f} GB free "
        f"({used_fraction:.2f} used)"
    )
    return dev_idx, free_gb, total_gb


def __parallel(  # ruff: ignore[too-many-arguments, too-many-locals, too-many-positional-arguments]
    models: list[Net],
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    trainer_kwargs: LTrainerKWArgs,
    max_gpu_memory: float,
    min_free_gpu_memory_gb: float | None = None,  # ruff: ignore[unused-function-argument]
    output_dir: Path | None = None,
) -> None:
    """Train models in parallel.

    Trains all models in parallel and schedule them such that the GPU
    memory is not exceeded. This schedulers requires an NVIDIA GPU.

    Args:
        models (list[Net]): List of models to train.
        train_dataloader (DataLoader): Data loader for the training
            data.
        test_dataloader (DataLoader): Data loader for the test data.
        trainer_kwargs (LTrainerKWArgs): Trainer keyword arguments.
        max_gpu_memory (float): Maximum GPU memory usage.
        min_free_gpu_memory_gb (float | None): Minimum free GPU memory.
        output_dir (Path | None): Optional directory to save training
            info.

    Raises:
        ImportError: If the NVIDIA Management Library is not found. This
            happens when the code is not running on an NVIDIA GPU.
        RuntimeError: If no NVIDIA GPUs are found.
    """
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError_LibraryNotFound as e:  # pragma: no cover  # type: ignore
        raise ImportError(
            "NVIDIA Management Library not found. Not running on an NVIDIA GPU."
        ) from e

    device_count = pynvml.nvmlDeviceGetCount()

    if device_count == 0:
        raise RuntimeError("No NVIDIA GPUs found")

    completed_jobs: list[SpawnProcess] = []
    results = mp.Manager().list()

    shared_dataset = SharedCASDataset(
        dataset=train_dataloader.dataset,  # type: ignore
        create=True,
    )

    # Create a PyTorch DataLoader for the dataset
    dataset = SharedCASDataset(
        dataset=shared_dataset,
        create=False,
    )
    shared_train_dataloader = DataLoader(
        dataset,
        batch_size=train_dataloader.batch_size,
        shuffle=True,
        num_workers=train_dataloader.num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=1,
    )
    shared_test_dataloader = DataLoader(
        dataset,
        batch_size=test_dataloader.batch_size,
        num_workers=test_dataloader.num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=1,
    )

    # Determine which GPUs to use
    all_device_indices = list(range(device_count))

    while models:
        # Check all GPUs
        gpu_status: list[tuple[int, float, float]] = []  # (idx, free_gb, total_gb)

        for dev_idx in all_device_indices:
            status = _query_gpu_status(dev_idx)
            if status is not None:
                gpu_status.append(status)

        # Find GPUs with sufficient memory (using GB-based threshold)
        # max_gpu_memory is fraction, convert to absolute GB for
        # comparison
        min_required_gb = max_gpu_memory * (gpu_status[0][2] if gpu_status else 0)

        available_gpus = [
            (idx, free, total)
            for idx, free, total in gpu_status
            if free >= min_required_gb
        ]

        if not available_gpus:
            logger.info(
                "No GPUs have sufficient memory. "
                f"Waiting for resources (min_required_gb={min_required_gb:.2f})"
            )
            time.sleep(5)
            continue

        # Use the GPU with the most free memory
        available_gpus.sort(key=operator.itemgetter(1), reverse=True)
        best_gpu = available_gpus[0]
        best_gpu_idx = best_gpu[0]
        best_gpu_free = best_gpu[1]

        # Start a new job on this GPU if we haven't exceeded device
        # count
        active_processes = sum(1 for p in completed_jobs if p.is_alive())

        if active_processes < device_count and models:
            model = models.pop(0)

            # Set CUDA_VISIBLE_DEVICES for this specific process
            original_env = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_idx)

            # Start a new process to train the model
            ctx = mp.get_context("spawn")
            process = ctx.Process(
                target=__train,
                args=(
                    model,
                    shared_train_dataloader,
                    shared_test_dataloader,
                    trainer_kwargs,
                    results,
                    output_dir,
                ),
            )
            process.start()
            completed_jobs.append(process)

            logger.info(
                f"Started training job for model on GPU {best_gpu_idx} "
                f"({best_gpu_free:.2f} GB free)"
            )

            # Restore original env
            if original_env is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_env
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            logger.debug(
                f"Waiting for GPU resources. Active processes: {active_processes}, "
                f"Models remaining: {len(models)}"
            )

        # Wait to ensure the model is loaded into memory or for the GPU
        # to free up memory.
        time.sleep(2)

    # Wait for all processes to complete
    for job in completed_jobs:
        job.join()


def train(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    models: list[Net],
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    trainer_kwargs: LTrainerKWArgs,
    strategy: Literal["sequential", "parallel"] = "sequential",
    max_gpu_memory: float = 0.8,
    min_free_gpu_memory_gb: float | None = None,
    output_dir: Path | None = None,
) -> None:
    """Train models using the specified strategy.

    The strategy can be either "sequential" or "parallel". The
    "sequential" strategy trains the models one after another, while the
    "parallel" strategy trains all models in parallel and schedules them
    based on the GPU memory usage.

    Args:
        models (list[Net]): List of models to train.
        train_dataloader (DataLoader): Data loader for the training
            data.
        test_dataloader (DataLoader): Data loader for the test data.
        trainer_kwargs (LTrainerKWArgs): Trainer keyword arguments.
        strategy (Literal["sequential", "parallel"]): Training
            strategy. Defaults to "sequential".
        max_gpu_memory (float): Maximum GPU memory usage (fraction 0-1).
            Defaults to 0.8.
        min_free_gpu_memory_gb (float | None): Minimum free GPU memory
            in GB to start a job. If None, uses max_gpu_memory as
            threshold. Defaults to None.
        output_dir (Path | None): Optional directory to save training
            info. Defaults to None.

    Raises:
        ValueError: If an invalid strategy is provided.
    """
    if strategy == "sequential":
        __sequential(
            models, train_dataloader, test_dataloader, trainer_kwargs, output_dir
        )
    elif strategy == "parallel":
        __parallel(
            models,
            train_dataloader,
            test_dataloader,
            trainer_kwargs,
            max_gpu_memory,
            min_free_gpu_memory_gb,
            output_dir,
        )
    else:
        raise ValueError(f"Invalid strategy: {strategy}")
