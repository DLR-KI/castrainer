# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""SafetyNet command: Create full safety net with all JSON files."""

from pathlib import Path
from typing import Annotated

import lightning as L  # ruff: ignore[lowercase-imported-as-non-lowercase]
import typer
from loguru import logger

from castrainer.safetynet.trainer import (
    DATA_DIR_ENV_VAR,
    SafetyNetTrainer,
    TrainingConfig,
    get_activation_fn,
)

app = typer.Typer(
    help="Create a full safety net (SafetyNet) for VCAS and HCAS systems",
    context_settings={"allow_interspersed_args": True},
)


def _create_training_config(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    activations: list[str] | None,
    hidden_nodes: list[int] | None,
    hidden_layers: int | None,
    batch_size: int | None,
    max_epochs: int | None,
    patience: int | None,
    nproc: int | None,
    enable_checkpointing: bool,  # ruff: ignore[boolean-type-hint-positional-argument]
    enable_progress_bar: bool,  # ruff: ignore[boolean-type-hint-positional-argument]
    one_hot: bool,  # ruff: ignore[boolean-type-hint-positional-argument]
) -> TrainingConfig:
    """Create training configuration from CLI arguments.

    Args:
        activations (list[str] | None): List of activation functions to
            use.
        hidden_nodes (list[int] | None): List of hidden nodes to use.
        hidden_layers (int | None): Number of hidden layers to use.
        batch_size (int | None): Batch size for training.
        max_epochs (int | None): Maximum number of epochs.
        patience (int | None): Early stopping patience.
        nproc (int | None): Number of processes for data loading.
        enable_checkpointing (bool): Enable model checkpointing.
        enable_progress_bar (bool): Enable progress bar.
        one_hot (bool): Use one-hot encoding for targets.

    Returns:
        TrainingConfig: Training configuration dataclass instance.
    """
    return TrainingConfig(
        activations=activations or ["relu"],
        hidden_nodes=hidden_nodes or [100],
        hidden_layers=[hidden_layers] if hidden_layers is not None else [4],
        batch_size=batch_size or 32,
        max_epochs=max_epochs or 10000,
        patience=patience or 1000,
        nproc=nproc or 8,
        enable_checkpointing=enable_checkpointing,
        enable_progress_bar=enable_progress_bar,
        one_hot=one_hot,
    )


@app.callback(invoke_without_command=True)
def safetynet(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    system: Annotated[
        list[str],
        typer.Argument(help="System(s) to train: vcas, hcas, or all"),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(help="Output directory for trained models"),
    ] = Path("safetynet"),
    data_dir: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Directory containing the generated HorizontalCAS/ and "
                "VerticalCAS/ training data. Defaults to "
                f"${DATA_DIR_ENV_VAR} if set, otherwise the current "
                "working directory."
            )
        ),
    ] = None,
    activations: Annotated[
        list[str] | None,
        typer.Option(help="Activation functions to use"),
    ] = None,
    hidden_nodes: Annotated[
        list[int] | None,
        typer.Option(help="Number of hidden nodes"),
    ] = None,
    hidden_layers: Annotated[
        int | None,
        typer.Option(help="Number of hidden layers"),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option(help="Batch size for training"),
    ] = None,
    max_epochs: Annotated[
        int | None,
        typer.Option(help="Maximum number of epochs"),
    ] = None,
    patience: Annotated[
        int | None,
        typer.Option(help="Early stopping patience"),
    ] = None,
    enable_progress_bar: Annotated[  # ruff: ignore[boolean-default-value-positional-argument]
        bool,
        typer.Option(
            "--enable-progress-bar/--no-progress-bar", help="Enable progress bar"
        ),
    ] = True,
    enable_checkpointing: Annotated[  # ruff: ignore[boolean-default-value-positional-argument]
        bool,
        typer.Option(
            "--enable-checkpointing/--no-checkpointing",
            help="Enable model checkpointing",
        ),
    ] = False,
    one_hot: Annotated[  # ruff: ignore[boolean-default-value-positional-argument]
        bool,
        typer.Option("--one-hot/--no-one-hot", help="Use one-hot encoding for targets"),
    ] = False,
    nproc: Annotated[
        int | None,
        typer.Option(help="Number of processes for data loading"),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(
            help=(
                "Random seed for reproducible training. Guarantees "
                "reproducibility only in combination with the (default) "
                "sequential strategy."
            )
        ),
    ] = None,
) -> None:
    """Create a full safety net including all JSON files.

    Args:
        system (list[str]): List of systems to train
            (vcas, hcas, or all).
        output_dir (Path): Output directory for trained models.
        data_dir (Path | None): Directory containing the generated
            HorizontalCAS/ and VerticalCAS/ training data. If None, the
            CASTRAINER_DATA_DIR environment variable is used, falling
            back to the current working directory.
        activations (list[str] | None): List of activation functions to
            use. If None, defaults to ["relu"].
        hidden_nodes (list[int] | None): List of hidden nodes to use.
            If None, defaults to [100].
        hidden_layers (int | None): Number of hidden layers to use. If
            None, defaults to 4.
        batch_size (int | None): Batch size for training. If None,
            defaults to 32.
        max_epochs (int | None): Maximum number of epochs. If None,
            defaults to 10000.
        patience (int | None): Early stopping patience. If None,
            defaults to 1000.
        enable_progress_bar (bool): Show the progress bar if True.
        enable_checkpointing (bool): Enable model checkpointing if True.
        one_hot (bool): Use one-hot encoding for targets if True.
        nproc (int | None): Number of processes for data loading. If
            None, defaults to 8.
        seed (int | None): Random seed for reproducible training. If
            None, training is non-deterministic.
    """
    if seed is not None:
        L.seed_everything(seed, workers=True)

    # Create training configuration
    config = _create_training_config(
        activations=activations,
        hidden_nodes=hidden_nodes,
        hidden_layers=hidden_layers,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        nproc=nproc,
        enable_checkpointing=enable_checkpointing,
        enable_progress_bar=enable_progress_bar,
        one_hot=one_hot,
    )

    # safetynet trains one architecture per system, not a sweep - if
    # more than one value was given for these, only the first is used
    # (`castrainer study` is the tool for sweeping hyperparameters).
    if len(config.activations) > 1:
        logger.warning(
            f"Multiple activations given ({config.activations}); safetynet "
            f"trains one model per system, using '{config.activations[0]}'. "
            "Use `castrainer study` to sweep activation functions."
        )
    if len(config.hidden_nodes) > 1:
        logger.warning(
            f"Multiple hidden-node counts given ({config.hidden_nodes}); "
            f"safetynet trains one model per system, using "
            f"{config.hidden_nodes[0]}. Use `castrainer study` to sweep "
            "hidden-node counts."
        )

    # Create trainer
    trainer = SafetyNetTrainer(config=config, output_dir=output_dir, data_dir=data_dir)

    # Determine systems to train
    systems: list[str] = []
    valid_systems = {"vcas", "hcas"}
    for sys_name in system:
        if sys_name == "all":
            systems.extend(["vcas", "hcas"])
        elif sys_name in valid_systems:
            systems.append(sys_name)
        else:
            typer.echo(f"Unknown system: {sys_name}", err=True)
            continue

    # Train each system
    for sys_name in systems:
        activation_fn = get_activation_fn(config.activations[0])
        typer.echo(f"# Training {sys_name.upper()}")
        trainer.train_system(system=sys_name, activation_fn=activation_fn)

    typer.echo("All trainings completed!\nEach system has its own manifest file:")
    for sys_name in systems:
        typer.echo(f"  - {output_dir}/{sys_name}/{sys_name}.json")
