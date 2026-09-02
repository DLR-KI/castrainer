# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Train a single neural network and corresponding LUT."""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import lightning as L  # ruff: ignore[lowercase-imported-as-non-lowercase]
import typer
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


app = typer.Typer(
    help="Train a single neural network and corresponding LUT",
    context_settings={"allow_interspersed_args": True},
)


@app.callback(invoke_without_command=True)
def train(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    file: Annotated[
        str,
        typer.Argument(help="HDF5 file for training"),
    ],
    activation: Annotated[
        str,
        typer.Option(help="Activation function (relu, leakyrelu, gelu, tanh, sigmoid)"),
    ] = "relu",
    hidden_nodes: Annotated[
        int,
        typer.Option(help="Number of hidden nodes per layer"),
    ] = 100,
    hidden_layers: Annotated[
        int,
        typer.Option(help="Number of hidden layers"),
    ] = 4,
    batch_size: Annotated[int, typer.Option(help="Batch size for training")] = 32,
    max_epochs: Annotated[int, typer.Option(help="Maximum epochs")] = 10_000,
    patience: Annotated[int, typer.Option(help="Early stopping patience")] = 1_000,
    nproc: Annotated[
        int, typer.Option(help="Number of processes for data loading")
    ] = 8,
    one_hot: Annotated[  # ruff: ignore[boolean-default-value-positional-argument]
        bool,
        typer.Option("--one-hot/--no-one-hot", help="Use one-hot encoding for targets"),
    ] = False,
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
    output_dir: Annotated[
        Path,
        typer.Option(help="Output directory for trained model"),
    ] = Path("trained"),
    seed: Annotated[
        int | None,
        typer.Option(help="Random seed for reproducible training"),
    ] = None,
) -> None:
    """Train a single neural network and corresponding LUT.

    Args:
        file (str): HDF5 file for training.
        activation (str): Activation function to use (relu, leakyrelu,
            gelu, tanh, sigmoid).
        hidden_nodes (int): Number of hidden nodes per layer.
        hidden_layers (int): Number of hidden layers.
        batch_size (int): Batch size for training.
        max_epochs (int): Maximum number of epochs to train.
        patience (int): Early stopping patience in epochs.
        nproc (int): Number of processes for data loading.
        one_hot (bool): Whether to use one-hot encoding for targets.
        enable_checkpointing (bool): Whether to enable model
            checkpointing.
        enable_progress_bar (bool): Whether to enable the progress bar
            during training.
        output_dir (Path): Directory where the trained model will be
            saved.
        seed (int | None): Random seed for reproducible training. If
            None, training is non-deterministic.
    """
    if seed is not None:
        L.seed_everything(seed, workers=True)

    # Set up thread limits
    set_thread_limits(nproc)

    file_path = Path(file)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return

    logger.info(f"Training on: {file_path}")

    # Get activation function
    activation_map: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    activation_fn = activation_map.get(activation.lower(), nn.ReLU)
    if activation.lower() not in activation_map:
        logger.warning(f"Unknown activation '{activation}', using ReLU")

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

    # Create model
    model = Net(
        inputs=len(dataset.x[0]),
        outputs=len(dataset.y[0]),
        hidden_layers=[hidden_nodes] * hidden_layers,
        activation=activation_fn,
        one_hot=one_hot,
    )

    # Setup trainer kwargs
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

    # Train model
    logger.info("Starting training...")

    train_models(
        models=[model],
        trainer_kwargs=trainer_kwargs,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        strategy="sequential",
        output_dir=output_dir,
    )

    logger.info(f"Training complete. Model saved to {output_dir}")
