# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""SafetyNet trainer for VCAS and HCAS systems.

Trainer methods require self, train_system is complex.
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from gc import get_referents
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Any, cast

import lightning as L  # ruff: ignore[lowercase-imported-as-non-lowercase]
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from castrainer.safetynet.config import HCAS_CONFIG, VCAS_CONFIG
from castrainer.train.data import CASDataset
from castrainer.train.lut import KDTreeLUT
from castrainer.train.net import Net
from castrainer.train.safetynet import SafetyNet

IGNORELIST = type, ModuleType, FunctionType
"""Object types to ignore when calculating size of an object."""

DATA_DIR_ENV_VAR = "CASTRAINER_DATA_DIR"
"""Environment variable pointing at the training data directory.

DATA_DIR_ENV_VAR is used when no explicit ``data_dir`` is given (handy
for batch schedulers where passing CLI flags through job scripts is
awkward).
"""


def resolve_data_dir(data_dir: Path | None = None) -> Path:
    """Resolve the directory holding the HDF5 training data.

    The training data is generated from the Julia MDP tables and is far
    too large to ship with the package, so its location is resolved at
    runtime, in this order:

    1. the explicit `data_dir` argument,
    2. the ``CASTRAINER_DATA_DIR`` environment variable,
    3. the current working directory (the repository root, when
        castrainer is run from a checkout).

    The resolved directory is expected to contain the
    ``HorizontalCAS/GenerateNetworks/`` and
    ``VerticalCAS/GenerateNetworks/`` subdirectories the generator
    scripts write their HDF5 files into.

    Args:
        data_dir (Path | None): Explicit training data directory, if
            any.

    Returns:
        Path: The resolved training data directory.
    """
    if data_dir is not None:
        return data_dir
    env_data_dir = os.environ.get(DATA_DIR_ENV_VAR)
    if env_data_dir:
        return Path(env_data_dir)
    return Path.cwd()


def _tensor_storage_bytes(tensor: torch.Tensor) -> int:
    """Best-effort estimate of a tensor's underlying storage size.

    Falls back across torch storage APIs, which have changed across
    versions. Returns 0 rather than raising if none are usable.

    Args:
        tensor (torch.Tensor): Tensor to measure.

    Returns:
        int: Size of the tensor's underlying storage in bytes.
    """
    try:
        return int(tensor.untyped_storage().nbytes())
    except Exception:  # ruff: ignore[blind-except, try-except-pass]
        pass
    try:
        storage = tensor.storage()
    except Exception:  # ruff: ignore[blind-except]
        return 0
    if hasattr(storage, "nbytes"):
        return int(storage.nbytes)
    return int(storage.size() * storage.element_size())


def _getsize(obj: object) -> int:
    """Get the true size of an object in bytes.

    Args:
        obj (object): Object to get the size of.

    Returns:
        int: Size of the object in bytes.

    Raises:
        TypeError: If the object type is in the ignore list.
    """
    if isinstance(obj, IGNORELIST):
        raise TypeError("getsize() does not take argument of type: " + str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj_ in objects:
            if not isinstance(obj_, IGNORELIST) and id(obj_) not in seen_ids:
                seen_ids.add(id(obj_))
                # Special-case torch tensors to include underlying
                # storage.
                if isinstance(obj_, torch.Tensor):
                    size += _tensor_storage_bytes(obj_)

                size += sys.getsizeof(obj_)
                need_referents.append(obj_)
        objects = get_referents(*need_referents)
    return size


@dataclass
class TrainingConfig:
    """Training configuration for SafetyNet.

    Attributes:
        activations (list[str]): List of activation functions to use.
        hidden_nodes (list[int]): Number of hidden nodes per layer.
        hidden_layers (list[int]): Number of hidden layers.
        batch_size (int): Batch size for training.
        max_epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        enable_checkpointing (bool): Whether to enable model
            checkpointing.
        enable_progress_bar (bool): Whether to show progress bar.
        one_hot (bool): Whether to use one-hot encoding for targets.
        nproc (int): Number of processes for data loading.
    """

    activations: list[str] = field(default_factory=lambda: ["relu"])
    hidden_nodes: list[int] = field(default_factory=lambda: [100])
    hidden_layers: list[int] = field(default_factory=lambda: [4])
    batch_size: int = 32
    max_epochs: int = 10000
    patience: int = 1000
    enable_checkpointing: bool = False
    enable_progress_bar: bool = True
    one_hot: bool = False
    nproc: int = 8


def get_activation_fn(name: str) -> type[nn.Module]:
    """Get activation function class from name.

    Args:
        name (str): Name of activation function.

    Returns:
        type[nn.Module]: Activation function class.
    """
    activation_map: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    return activation_map.get(name.lower(), nn.ReLU)


class SafetyNetTrainer:
    """Trainer for SafetyNet systems (VCAS and HCAS).

    Attributes:
        config (TrainingConfig): Training configuration.
        output_dir (Path): Output directory for trained models.
    """

    def __init__(
        self,
        config: TrainingConfig | None = None,
        output_dir: Path | None = None,
        data_dir: Path | None = None,
    ) -> None:
        """Initialize the SafetyNet trainer.

        Args:
            config (TrainingConfig | None): Training configuration. If
                None, uses defaults.
            output_dir (Path | None): Output directory for trained
                models.
            data_dir (Path | None): Directory holding the generated
                HorizontalCAS/VerticalCAS training data. If None, it is
                resolved via `resolve_data_dir`.
        """
        self.config = config or TrainingConfig()
        self.output_dir = output_dir or Path("safetynet")
        self.data_dir = resolve_data_dir(data_dir)

    def _get_data_path(self, system: str, subsystem_id: int | None = None) -> Path:
        """Get path to training data file.

        Args:
            system (str): System name ("vcas" or "hcas").
            subsystem_id (int | None): Subsystem ID or None for
                directory.

        Returns:
            Path: Path to data file or directory.

        Raises:
            ValueError: If system is unknown.
        """
        # The HDF5 training data is generated from the Julia MDP tables
        # (several GB) and is therefore not shipped with the package.
        # It is looked up below `self.data_dir`, which keeps the layout
        # of the repository the generator scripts write into.
        data_dir = {
            "vcas": "VerticalCAS/GenerateNetworks",
            "hcas": "HorizontalCAS/GenerateNetworks",
        }.get(system)

        if data_dir is None:
            raise ValueError(f"Unknown system: {system}")

        if subsystem_id is None:
            return self.data_dir / data_dir

        if system == "vcas":
            filename = f"VCAS_TrainingData_v5_{subsystem_id:02d}.h5"
        else:
            # HCAS: 40 subsystems (5 pra x 8 tau)
            tau_values = [0, 5, 10, 15, 20, 30, 40, 60]
            pra = subsystem_id // 8
            tau_idx = subsystem_id % 8
            tau = tau_values[tau_idx]
            filename = f"HCAS_rect_TrainingData_v6_pra{pra}_tau{tau:02d}.h5"

        return self.data_dir / data_dir / filename

    def _create_model(
        self,
        inputs: int,
        outputs: int,
        activation_fn: type[nn.Module],
    ) -> Net:
        """Create a neural network model.

        Args:
            inputs (int): Number of input features.
            outputs (int): Number of outputs.
            activation_fn (type[nn.Module]): Activation function.

        Returns:
            Net: Neural network model.
        """
        return Net(
            inputs=inputs,
            outputs=outputs,
            hidden_layers=[self.config.hidden_nodes[0]] * self.config.hidden_layers[0],
            activation=activation_fn,
            one_hot=self.config.one_hot,
        )

    @staticmethod
    def _save_model(model: Net, output_path: Path) -> None:
        """Save trained model to file.

        Args:
            model (Net): Trained model.
            output_path (Path): Output file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)
        logger.info(f"Model saved to {output_path}")

    @staticmethod
    def _save_lut(
        lut: KDTreeLUT,
        output_path: Path,
        sys_config: dict[str, Any],
    ) -> None:
        """Save LUT to JSON file in safetynet.schema.json format.

        Args:
            lut (KDTreeLUT): Lookup table.
            output_path (Path): Output file path.
            sys_config (dict[str, Any]): System configuration.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert LUT to serializable format
        data = lut.to_serializable_entries()

        # Create LUT schema
        lut_schema = {
            "version": "1.0.0",
            "description": sys_config.get("description", "SafetyNet LUT"),
            "datatype": sys_config.get("datatype", "float32"),
            "inputs": sys_config.get("inputs", []),
            "numberOutputs": sys_config.get("num_outputs", 1),
            "data": data,
        }

        with Path(output_path).open("w", encoding="utf-8") as f:
            json.dump(lut_schema, f, indent=2)

        logger.info(f"LUT saved to {output_path} ({len(data)} entries)")

    @staticmethod
    def _generate_manifest(
        system: str,
        output_dir: Path,
        networks: list[dict[str, Any]],
        luts: list[dict[str, Any]],
        sys_config: dict[str, Any],
    ) -> Path:
        """Generate manifest file for the SafetyNet system.

        Args:
            system (str): System name.
            output_dir (Path): Output directory.
            networks (list[dict[str, Any]]): List of network file
                entries.
            luts (list[dict[str, Any]]): List of LUT file entries.
            sys_config (dict[str, Any]): System configuration.

        Returns:
            Path: Path to manifest file.
        """
        manifest = {
            "version": "1.0.0",
            "description": sys_config.get("description", ""),
            "function": sys_config.get("function", ""),
            "datatype": sys_config.get("datatype", "float32"),
            "inputs": sys_config.get("inputs", []),
            "numberOutputs": sys_config.get("num_outputs", 1),
            "networks": networks,
            "luts": luts,
        }

        manifest_path = output_dir / f"{system}.json"
        with Path(manifest_path).open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Manifest saved to {manifest_path}")
        return manifest_path

    def _save_training_info(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        self,
        model: Net,
        safetynet: SafetyNet,
        dataset: CASDataset,
        trainer: L.Trainer,
        subsystem_name: str,
        training_duration: float,
        output_dir: Path,
    ) -> None:
        """Save training info for a subsystem.

        Args:
            model (Net): Trained model.
            safetynet (SafetyNet): SafetyNet with filled LUT.
            dataset (CASDataset): Training dataset.
            trainer (L.Trainer): PyTorch Lightning trainer.
            subsystem_name (str): Name of the subsystem.
            training_duration (float): Training duration in seconds.
            output_dir (Path): Output directory for training info.
        """
        # Extract final loss from training results
        final_train_loss = None
        final_test_loss = None

        # Get training loss from trainer logs
        if (
            hasattr(trainer, "callback_metrics")
            and "train_loss" in trainer.callback_metrics
        ):
            final_train_loss = float(trainer.callback_metrics["train_loss"].item())

        # Get test loss from trainer results
        if hasattr(trainer, "logged_metrics"):
            for key, value in trainer.logged_metrics.items():
                if "test_loss" in key:
                    final_test_loss = (
                        float(value.item()) if hasattr(value, "item") else float(value)
                    )
                    break

        training_info = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "subsystem": subsystem_name,
            "hyperparameters": {
                "activation": str(model.activation()),
                "hidden_layers": model.hidden_layers,
                "num_hidden_layers": len(model.hidden_layers or []),
                "inputs": model.inputs,
                "outputs": model.outputs,
                "one_hot": self.config.one_hot,
                "batch_size": self.config.batch_size,
                "max_epochs": self.config.max_epochs,
                "patience": self.config.patience,
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
                "network_size_bytes": sum(
                    p.numel() * p.element_size() for p in model.parameters()
                ),
                "network_size_mb": sum(
                    p.numel() * p.element_size() for p in model.parameters()
                )
                / (1024 * 1024),
                "lut_size_bytes": safetynet.lut.size_bytes(),
                "lut_size_mb": safetynet.lut.size_bytes() / (1024 * 1024),
                "safety_net_total_bytes": sum(
                    p.numel() * p.element_size() for p in model.parameters()
                )
                + safetynet.lut.size_bytes(),
                "safety_net_total_mb": (
                    sum(p.numel() * p.element_size() for p in model.parameters())
                    + safetynet.lut.size_bytes()
                )
                / (1024 * 1024),
            },
            "lut_statistics": {
                "lut_entries": len(safetynet.lut),
                "dataset_size": len(dataset),
                "lut_coverage": len(safetynet.lut) / len(dataset),
                "nn_coverage": 1 - len(safetynet.lut) / len(dataset),
            },
            "dataset": {
                "filename": str(dataset.file)
                if hasattr(dataset, "file")
                else "unknown",
                "size": len(dataset),
            },
        }

        info_file = output_dir / f"{subsystem_name}_training_info.json"
        with info_file.open("w", encoding="utf-8") as f:
            json.dump(training_info, f, indent=2)

        logger.info(f"Training info saved to {info_file}")

    def train_system(  # ruff: ignore[too-many-locals, too-many-statements]
        self,
        system: str,
        activation_fn: type[nn.Module] = nn.ReLU,
    ) -> None:
        """Train all subsystems for a given system.

        Args:
            system (str): System name ("vcas" or "hcas").
            activation_fn (type[nn.Module]): Activation function to
                use.

        Raises:
            ValueError: If system is unknown.
        """
        sys_config = {"vcas": VCAS_CONFIG, "hcas": HCAS_CONFIG}.get(system)
        if sys_config is None:
            raise ValueError(f"Unknown system: {system}")

        subsystem_ids = list(range(sys_config["num_subsystems"]))
        if system == "vcas":
            subsystem_ids = list(range(1, 10))  # 1-9 for VCAS

        logger.info(f"Training {system.upper()} with {len(subsystem_ids)} subsystems")
        logger.info(f"Output directory: {self.output_dir / system}")

        output_dir = self.output_dir / system
        output_dir.mkdir(parents=True, exist_ok=True)

        networks: list[dict[str, Any]] = []
        luts: list[dict[str, Any]] = []

        # Train each subsystem
        for subsystem_id in subsystem_ids:
            # Create fresh trainer_kwargs and callbacks for EACH
            # subsystem. This ensures each subsystem has independent
            # early stopping with its own patience counter and best loss
            # tracking.
            subsystem_trainer_kwargs = {
                "max_epochs": self.config.max_epochs,
                "enable_progress_bar": self.config.enable_progress_bar,
                "enable_checkpointing": self.config.enable_checkpointing,
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        mode="min",
                        patience=self.config.patience,
                        verbose=False,
                    )
                ],
            }

            if system == "vcas":
                subsystem_name = f"{system}_{subsystem_id:02d}"
            else:
                tau_values = [0, 5, 10, 15, 20, 30, 40, 60]
                pra = subsystem_id // 8
                tau = tau_values[subsystem_id % 8]
                subsystem_name = f"{system}_pra{pra}_tau{tau:02d}"

            logger.info(f"\n{'=' * 60}")
            logger.info(
                f"Training subsystem {subsystem_id + 1}/{len(subsystem_ids)}: "
                f"{subsystem_name}"
            )
            logger.info(f"{'=' * 60}")

            # Get data path
            data_path = self._get_data_path(system, subsystem_id)

            if not data_path.exists():
                logger.error(
                    f"Data file not found: {data_path}. Generate the training "
                    f"data first (see README) or point castrainer at it with "
                    f"--data-dir / ${DATA_DIR_ENV_VAR}."
                )
                continue

            # Load dataset
            logger.info(f"Loading data from {data_path}")
            dataset = CASDataset(data_path, one_hot=self.config.one_hot)

            # Infer input/output dimensions from data
            num_inputs = dataset.x.shape[1]
            num_outputs = dataset.y.shape[1]
            logger.info(f"Data dimensions: inputs={num_inputs}, outputs={num_outputs}")

            # Create data loaders
            train_dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.nproc,
                persistent_workers=True,
                pin_memory=True,
            )
            test_dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.nproc,
                persistent_workers=True,
                pin_memory=True,
            )

            # Create model
            model = self._create_model(
                inputs=num_inputs,
                outputs=num_outputs,
                activation_fn=activation_fn,
            )

            # Train model
            logger.info("Starting training...")
            train_start_time = time.perf_counter()
            trainer = L.Trainer(**cast("Any", subsystem_trainer_kwargs))
            trainer.fit(model=model, train_dataloaders=train_dataloader)

            # Run test
            result = trainer.test(model=model, dataloaders=test_dataloader)
            logger.info(f"Test loss: {result[0]['test_loss']:.6f}")

            train_duration = time.perf_counter() - train_start_time

            # Save model
            model_path = output_dir / f"{subsystem_name}.pt"
            self._save_model(model, model_path)

            # Generate and save LUT
            logger.info("Generating LUT...")
            safetynet = SafetyNet(model)
            safetynet.fill(dataset)

            lut_path = output_dir / f"{subsystem_name}_lut.json"
            self._save_lut(safetynet.lut, lut_path, sys_config)

            # Save training info
            self._save_training_info(
                model=model,
                safetynet=safetynet,
                dataset=dataset,
                trainer=trainer,
                subsystem_name=subsystem_name,
                training_duration=train_duration,
                output_dir=output_dir,
            )

            # Build "if" conditions based on subsystem selection
            # parameters. Only include parameters that are relevant for
            # selecting this network/LUT
            if_condition: dict[str, Any] = {}

            if system == "vcas":
                # VCAS: subsystems are split by ra (range advisory
                # region 1-9). The ra corresponds to the subsystem_id
                if_condition["ra"] = {
                    "minimum": float(subsystem_id),
                    "maximum": float(subsystem_id),
                }
            elif system == "hcas":
                # HCAS: subsystems split by pra (0-4) and tau
                tau_values = [0, 5, 10, 15, 20, 30, 40, 60]
                pra = subsystem_id // 8
                tau = tau_values[subsystem_id % 8]

                if_condition["pra"] = {
                    "minimum": float(pra),
                    "maximum": float(pra),
                }
                if_condition["tau"] = {
                    "minimum": float(tau),
                    "maximum": float(tau),
                }

            # Add to manifest lists
            networks.append({
                "file": f"{subsystem_name}.pt",
                "networkFormat": "torch",
                "if": if_condition.copy(),
            })

            luts.append({
                "file": f"{subsystem_name}_lut.json",
                "lutFormat": "snet",
                "if": if_condition.copy(),
            })

        # Generate manifest
        logger.info(f"\n{'=' * 60}")
        logger.info("Generating manifest...")
        logger.info(f"{'=' * 60}")
        self._generate_manifest(system, output_dir, networks, luts, sys_config)
        logger.info(f"\n{'=' * 60}")
