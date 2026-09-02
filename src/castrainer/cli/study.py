# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Study command: Train models with hyperparameter study."""

import glob
import itertools
import json
import re
import time
from collections import namedtuple
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import lightning as L  # ruff: ignore[lowercase-imported-as-non-lowercase]
import torch
import typer
import yaml
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from castrainer.cli.utils import set_thread_limits
from castrainer.train.data import CASDataset
from castrainer.train.net import Net
from castrainer.train.train import train as train_models

if TYPE_CHECKING:
    from castrainer.train.train import LTrainerKWArgs

NetParameter = namedtuple(
    "NetParameter", ["activation_fn", "n_hidden_nodes", "n_hidden_layers"]
)

app = typer.Typer(
    help="Train models with hyperparameter study",
    context_settings={"allow_interspersed_args": True},
)


def _parse_comma_separated_list(value: str) -> list[str]:
    """Parse a comma-separated string into a list.

    Args:
        value (str): A string containing comma-separated values.

    Returns:
        list[str]: A list of strings, with whitespace stripped from each
            item.
    """
    return [item.strip() for item in value.split(",")]


def _expand_file_paths(file_patterns: list[str]) -> list[Path]:
    """Expand file patterns (glob) and directories into a list of paths.

    Args:
        file_patterns (list[str]): A list of file patterns, which can
            include glob patterns or directories.

    Returns:
        list[Path]: A list of Path objects matching the given patterns.
    """
    paths: list[Path] = []
    for pattern in file_patterns:
        p = Path(pattern)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.h5")))
        elif "*" in pattern or "?" in pattern:
            matched = glob.glob(pattern)  # ruff: ignore[glob]
            paths.extend(Path(m) for m in sorted(matched))
        elif p.exists():
            paths.append(p)
        else:
            logger.warning(f"No matching files found for: {pattern}")
    return paths


def _iter_hyperparameters(
    activations: list[str],
    hidden_nodes: list[int],
    hidden_layers: list[int],
) -> Iterator[NetParameter]:
    """Iterate over hyperparameter combinations.

    Args:
        activations (list[str]): A list of activation function names.
        hidden_nodes (list[int]): A list of integers representing the
            number of hidden nodes per layer.
        hidden_layers (list[int]): A list of integers representing the
            number of hidden layers.

    Yields:
        NetParameter: A named tuple containing the activation function,
            number of hidden nodes, and number of hidden layers for each
            combination.
    """
    activation_map: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }

    activation_fns = []
    for act in activations:
        if act.lower() in activation_map:
            activation_fns.append(activation_map[act.lower()])
        else:
            logger.warning(f"Unknown activation function: {act}, falling back to ReLU")
            activation_fns.append(nn.ReLU)

    yield from itertools.starmap(
        NetParameter,
        itertools.product(activation_fns, hidden_nodes, hidden_layers),
    )


def generate_hyperparameter_combinations(config: dict[str, Any]) -> list[NetParameter]:
    """Generate hyperparameter combinations from config.

    Args:
        config (dict[str, Any]): A dictionary containing lists of
            hyperparameters. Expected keys: "activations",
            "hidden_nodes", "hidden_layers".

    Returns:
        list[NetParameter]: A list of NetParameter named tuples for each
            combination of hyperparameters.
    """
    activations = config.get("activations", ["relu"])
    hidden_nodes = config.get("hidden_nodes", [25, 50, 100, 150, 200])
    hidden_layers = config.get("hidden_layers", [2, 3, 5, 7])
    return list(_iter_hyperparameters(activations, hidden_nodes, hidden_layers))


def get_results_dir(experiment_name: str) -> Path:
    """Get the results directory for an experiment.

    Args:
        experiment_name (str): The name of the experiment, used to
            organize results.

    Returns:
        Path: The path to the results directory for the experiment.
    """
    return (
        Path("results")
        / experiment_name
        / datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    )


def load_completed_combinations(results_dir: Path) -> set[str]:
    """Load completed combinations from status file.

    Args:
        results_dir (Path): The directory where the status file is
            located.

    Returns:
        set[str]: A set of identifiers for completed hyperparameter
            combinations, formatted as
            "activation_n{hidden_nodes}_l{hidden_layers}".
    """
    status_file = results_dir / "status.json"
    if not status_file.exists():
        return set()
    with status_file.open("r", encoding="utf-8") as f:
        status = json.load(f) or {}
    completed = set()
    for comb in status.get("completed_combinations", []):
        if isinstance(comb, dict):
            act_name = comb.get("activation", "").split(".")[-1]
            n_hidden_nodes = comb.get("n_hidden_nodes", 0)
            n_hidden_layers = comb.get("n_hidden_layers", 0)
            ident = f"{act_name}_n{n_hidden_nodes}_l{n_hidden_layers}"
            completed.add(ident)
    return completed


def save_completed_combinations(
    results_dir: Path, completed: list[dict[str, Any]]
) -> None:
    """Save completed combinations to status file.

    Args:
        results_dir (Path): The directory where the status file will be
            saved.
        completed (list[dict[str, Any]]): A list of dictionaries
            representing completed hyperparameter combinations, each
            containing keys "activation", "n_hidden_nodes", and
            "n_hidden_layers".
    """
    status_file = results_dir / "status.json"
    results_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "completed_combinations": completed,
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "num_completed": len(completed),
    }
    with status_file.open("w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)


def get_available_devices(devices_str: str) -> list[int]:
    """Get list of available GPU devices.

    Args:
        devices_str (str): A comma-separated string of GPU device
            indices to monitor (e.g. "0,1,2"). If empty, no devices will
            be monitored.

    Returns:
        list[int]: A list of available GPU device indices from the
            provided list. If a device is not available, it will be
            skipped with a warning.
    """
    if not devices_str.strip():
        return []
    devices = [int(d.strip()) for d in devices_str.split(",") if d.strip()]
    available_devices = []
    for dev in devices:
        if torch.cuda.is_available() and torch.cuda.device_count() > dev:
            available_devices.append(dev)
        else:
            logger.warning(f"GPU device {dev} not available, skipping")
    return available_devices


def _gpu_meets_memory_threshold(dev: int, min_free_memory_gb: float) -> bool:
    """Check whether one GPU device has enough free memory.

    Args:
        dev (int): GPU device index.
        min_free_memory_gb (float): Minimum free memory required, in
            GB.

    Returns:
        bool: True if the device has at least `min_free_memory_gb`
            free.
    """
    import pynvml  # ruff: ignore[import-outside-top-level]

    handle = pynvml.nvmlDeviceGetHandleByIndex(dev)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    free_gb = info.free / 1024**3
    logger.debug(f"GPU {dev}: {free_gb:.2f} GB free")
    if free_gb < min_free_memory_gb:
        logger.warning(
            f"GPU {dev}: {free_gb:.2f} GB < {min_free_memory_gb} GB required"
        )
        return False
    return True


def check_gpu_memory(devices: list[int], min_free_memory_gb: float) -> bool:
    """Check if GPU has sufficient free memory.

    Args:
        devices (list[int]): A list of GPU device indices to check.
        min_free_memory_gb (float): The minimum free memory in GB
            required to consider the GPU suitable for training.

    Returns:
        bool: True if all specified GPUs have at least the minimum free
            memory, False otherwise.
    """
    try:
        import pynvml  # ruff: ignore[import-outside-top-level]

        pynvml.nvmlInit()
        return all(
            _gpu_meets_memory_threshold(dev, min_free_memory_gb) for dev in devices
        )
    except ImportError:
        logger.error("pynvml not available, cannot check GPU memory")
        return False
    except Exception:  # ruff: ignore[blind-except]
        logger.error("Error checking GPU memory")
        return False


def load_hyperparameter_config(config_path: str) -> dict[str, Any]:
    """Load hyperparameter configuration from YAML/JSON file.

    Args:
        config_path (str): The path to the YAML or JSON configuration
            file containing hyperparameter lists.

    Returns:
        dict[str, Any]: A dictionary containing the hyperparameter lists
            loaded from the file. Expected keys include "activations",
            "hidden_nodes", and "hidden_layers".

    Raises:
        ValueError: If the config file does not exist or cannot be
            parsed.
    """
    path = Path(config_path)
    if not path.exists():
        raise ValueError(f"Config file not found: {config_path}")
    with path.open("r", encoding="utf-8") as f:
        if config_path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        return json.load(f)


@app.callback(invoke_without_command=True)
def study(  # ruff: ignore[complex-structure, too-many-branches, too-many-arguments, too-many-locals, too-many-statements, too-many-positional-arguments]
    file: Annotated[
        list[str],
        typer.Argument(
            help="HDF5 file(s) or directory path(s). Supports glob patterns like *.h5"
        ),
    ],
    nproc: Annotated[
        int, typer.Option(help="Number of processes for data loading")
    ] = 8,
    one_hot: Annotated[  # ruff: ignore[boolean-default-value-positional-argument]
        bool,
        typer.Option("--one-hot/--no-one-hot", help="Use one-hot encoding for targets"),
    ] = True,
    strategy: Annotated[
        str,
        typer.Option(help="Training strategy"),
    ] = "sequential",
    min_free_gpu_memory: Annotated[
        float, typer.Option(help="Minimum free GPU memory (GB) required to start a job")
    ] = 3.0,
    devices: Annotated[
        str, typer.Option(help="GPU devices to monitor (comma-separated)")
    ] = "0",
    experiment_name: Annotated[
        str, typer.Option(help="Experiment name for result organization")
    ] = "default",
    resume: Annotated[  # ruff: ignore[boolean-default-value-positional-argument]
        bool, typer.Option(help="Skip already completed hyperparameter combinations")
    ] = False,
    activations: Annotated[
        str,
        typer.Option(help="Comma-separated activation functions"),
    ] = "relu,leakyrelu,gelu",
    hidden_nodes: Annotated[
        str, typer.Option(help="Comma-separated hidden node counts")
    ] = "25,50,100,150,200",
    hidden_layers: Annotated[
        str, typer.Option(help="Comma-separated hidden layer counts")
    ] = "2,3,5,7",
    batch_size: Annotated[int, typer.Option(help="Batch size for training")] = 32,
    max_epochs: Annotated[int, typer.Option(help="Maximum epochs")] = 10_000,
    patience: Annotated[int, typer.Option(help="Early stopping patience")] = 1_000,
    enable_checkpointing: Annotated[  # ruff: ignore[boolean-default-value-positional-argument]
        bool,
        typer.Option(
            "--enable-checkpointing/--no-checkpointing",
            help="Enable model checkpointing",
        ),
    ] = False,
    enable_progress_bar: Annotated[  # ruff: ignore[boolean-default-value-positional-argument]
        bool,
        typer.Option(
            "--enable-progress-bar/--no-progress-bar", help="Enable progress bar"
        ),
    ] = True,
    config_file: Annotated[
        str | None, typer.Option(help="YAML/JSON config file for hyperparameters")
    ] = None,
    num_jobs: Annotated[
        int | None, typer.Option(help="Number of job chunks to run")
    ] = None,
    job_id: Annotated[
        int | None, typer.Option(help="Job chunk ID (0-based) for parallelization")
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(
            help=(
                "Random seed for reproducible training. Guarantees "
                "reproducibility only in combination with "
                "--strategy sequential."
            )
        ),
    ] = None,
) -> None:
    """Train models on the given dataset(s) with hyperparameter study.

    Args:
        file (list[str]): A list of HDF5 file paths or directory paths
            containing the training data. Supports glob patterns like
            "*.h5".
        nproc (int): The number of processes to use for data loading.
        one_hot (bool): Whether to use one-hot encoding for targets.
        strategy (str): The training strategy to use ("sequential" or
            "parallel").
        min_free_gpu_memory (float): Minimum free GPU memory in GB
            required to start a job.
        devices (str): Comma-separated list of GPU device indices to
            monitor (e.g. "0,1,2").
        experiment_name (str): The name of the experiment for organizing
            results.
        resume (bool): Whether to skip already completed hyperparameter
            combinations.
        activations (str): Comma-separated activation functions to use
            in the study.
        hidden_nodes (str): Comma-separated hidden node counts to use in
            the study.
        hidden_layers (str): Comma-separated hidden layer counts to use
            in the study.
        batch_size (int): Batch size for training.
        max_epochs (int): Maximum number of epochs for training.
        patience (int): Early stopping patience in epochs.
        enable_checkpointing (bool): Whether to enable model
            checkpointing during training.
        enable_progress_bar (bool): Whether to enable the progress bar
            during training.
        config_file (str | None): Path to a YAML or JSON configuration
            file containing hyperparameter lists. If provided, this will
            override the corresponding command-line options for
            activations, hidden nodes, and hidden layers.
        num_jobs (int | None): If provided, the total number of job
            chunks to split the hyperparameter combinations into for
            parallel execution. If not provided, all combinations will
            be run in a single job.
        job_id (int | None): If num_jobs is provided, this specifies the
            0-based ID of the current job chunk to run. For example, if
            num_jobs=4 and job_id=1, this job will run the second
            quarter of the hyperparameter combinations.
        seed (int | None): Random seed for reproducible training. If
            None, training is non-deterministic.
    """
    if seed is not None:
        L.seed_everything(seed, workers=True)

    # Validate strategy
    if strategy not in {"sequential", "parallel"}:
        logger.error(f"Invalid strategy: {strategy}")
        return

    # Set up thread limits
    set_thread_limits(nproc)

    # Get results directory
    results_dir = get_results_dir(experiment_name)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load config file if provided
    hyperparameter_config: dict[str, Any] | None = None
    if config_file:
        hyperparameter_config = load_hyperparameter_config(config_file)
        if activations != "relu,leakyrelu,gelu":
            hyperparameter_config["activations"] = _parse_comma_separated_list(
                activations
            )
        if hidden_nodes != "25,50,100,150,200":
            hyperparameter_config["hidden_nodes"] = [
                int(x) for x in _parse_comma_separated_list(hidden_nodes)
            ]
        if hidden_layers != "2,3,5,7":
            hyperparameter_config["hidden_layers"] = [
                int(x) for x in _parse_comma_separated_list(hidden_layers)
            ]
    else:
        hyperparameter_config = {
            "activations": _parse_comma_separated_list(activations),
            "hidden_nodes": [int(x) for x in _parse_comma_separated_list(hidden_nodes)],
            "hidden_layers": [
                int(x) for x in _parse_comma_separated_list(hidden_layers)
            ],
        }

    # Generate hyperparameter combinations
    all_combinations = generate_hyperparameter_combinations(hyperparameter_config)

    # Handle job splitting
    if num_jobs is not None and job_id is not None:
        chunk_size = (len(all_combinations) + num_jobs - 1) // num_jobs
        start_idx = job_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(all_combinations))
        combinations = all_combinations[start_idx:end_idx]
        logger.info(
            f"Running job {job_id}/{num_jobs}: combinations {start_idx}-{end_idx} "
            f"of {len(all_combinations)}"
        )
    else:
        combinations = all_combinations

    # Load completed combinations for resume
    completed_identifiers: set[str] = set()
    if resume:
        completed_identifiers = load_completed_combinations(results_dir)
        logger.info(f"Found {len(completed_identifiers)} completed combinations")

    # Filter out completed combinations
    if resume and completed_identifiers:
        filtered_combinations = []
        for combo in combinations:
            act_name = combo.activation_fn.__name__.lower()
            ident = f"{act_name}_n{combo.n_hidden_nodes}_l{combo.n_hidden_layers}"
            if ident not in completed_identifiers:
                filtered_combinations.append(combo)
            else:
                logger.debug(f"Skipping completed combination: {ident}")
        combinations = filtered_combinations

    # Expand file paths
    file_paths = _expand_file_paths(file)

    if not file_paths:
        logger.error("No input files found")
        return

    logger.info(f"Processing {len(file_paths)} file(s)")

    # Process each file
    for file_path in file_paths:
        logger.info(f"Training on: {file_path}")

        dataset = CASDataset(file_path, one_hot=one_hot)

        # Create data loaders
        train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=nproc,
            persistent_workers=True,
            pin_memory=True,
        )
        test_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=nproc,
            persistent_workers=True,
            pin_memory=True,
        )

        # Create models
        models: list[Net] = []
        for combo in combinations:
            net = Net(
                inputs=len(dataset.x[0]),
                outputs=len(dataset.y[0]),
                hidden_layers=[combo.n_hidden_nodes] * combo.n_hidden_layers,
                activation=combo.activation_fn,
                one_hot=one_hot,
            )
            models.append(net)

        # Check GPU memory for parallel strategy
        if strategy == "parallel":
            available_devices = get_available_devices(devices)
            if not available_devices:
                logger.error("No available GPUs, switching to sequential strategy")
                strategy = "sequential"
            elif not check_gpu_memory(available_devices, min_free_gpu_memory):
                logger.error(
                    "Insufficient GPU memory, switching to sequential strategy"
                )
                strategy = "sequential"

        trainer_kwargs: LTrainerKWArgs = {
            "max_epochs": max_epochs,
            "enable_progress_bar": enable_progress_bar,
            "enable_checkpointing": enable_checkpointing,
            "callbacks": [
                EarlyStopping(
                    monitor="train_loss",
                    mode="min",
                    patience=patience,
                    verbose=True,
                )
            ],
        }

        # Train models
        start_time = time.perf_counter()

        train_models(
            models=models,
            trainer_kwargs=trainer_kwargs,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            strategy=strategy,
            max_gpu_memory=min_free_gpu_memory,
            min_free_gpu_memory_gb=min_free_gpu_memory,
            output_dir=None,
        )

        end_time = time.perf_counter()
        logger.info(f"Training time: {end_time - start_time:.2f} seconds")

        # Collect results
        for combo in combinations:
            act_name = combo.activation_fn.__name__.lower()
            ident = f"{act_name}_n{combo.n_hidden_nodes}_l{combo.n_hidden_layers}"
            completed_identifiers.add(ident)

        # Save completed combinations
        completed_list = []
        for ident in completed_identifiers:
            match = re.match(r"(\w+)_n(\d+)_l(\d+)", ident)
            if match:
                completed_list.append({
                    "activation": match.group(1),
                    "n_hidden_nodes": int(match.group(2)),
                    "n_hidden_layers": int(match.group(3)),
                })

        save_completed_combinations(results_dir, completed_list)
        logger.info(f"Completed {len(completed_identifiers)} combinations")

    logger.info(f"Training complete. Results saved to {results_dir}")
