# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Load command: Load SafetyNet from JSON files into memory."""

from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from castrainer.safetynet.io import load_safetynet_directory
from castrainer.train.lut import KDTreeLUT
from castrainer.train.net import Net

app = typer.Typer(
    help="Load SafetyNet from JSON files into memory",
    context_settings={"allow_interspersed_args": True},
)


@app.callback(invoke_without_command=True)
def load(
    safetynet_dir: Annotated[
        Path,
        typer.Argument(help="Directory containing SafetyNet files"),
    ],
    system: Annotated[
        str | None,
        typer.Option(
            help=(
                "System to load (vcas or hcas). If None, discovers all "
                "available systems."
            )
        ),
    ] = None,
) -> tuple[dict[str, list[Net]], dict[str, list[KDTreeLUT]]]:
    """Load SafetyNet from JSON files into memory.

    Args:
        safetynet_dir (Path): Directory containing the SafetyNet JSON
            files.
        system (str | None): Optional system to load (vcas or hcas).
            If None, discovers all available systems.

    Returns:
        tuple[dict[str, list[Net]], dict[str, list[KDTreeLUT]]]: A
            tuple containing the loaded networks and LUTs.

    Raises:
        Exit: If the specified directory does not exist, if no
            valid systems are found, or if the specified system is not
            available.
    """
    logger.info(f"Loading SafetyNet from: {safetynet_dir}")
    try:
        loaded = load_safetynet_directory(safetynet_dir, system)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc

    loaded_networks = {
        sys_name: [bundle.model for bundle in bundles]
        for sys_name, bundles in loaded.items()
    }
    loaded_luts = {
        sys_name: [bundle.safetynet.lut for bundle in bundles]
        for sys_name, bundles in loaded.items()
    }

    typer.echo("SafetyNet loaded successfully")
    for sys_name, bundles in loaded.items():
        total_entries = sum(len(bundle.safetynet.lut) for bundle in bundles)
        total_size = sum(bundle.safetynet.lut.size_bytes() for bundle in bundles)
        typer.echo(
            f"  {sys_name.upper()}: {len(bundles)} networks, "
            f"{total_entries} LUT entries, {total_size} bytes via k-d tree"
        )

    return loaded_networks, loaded_luts
