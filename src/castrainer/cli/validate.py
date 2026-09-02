# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Validate SafetyNet directory structure and files."""

from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from castrainer.resources import (
    MANIFEST_SCHEMA,
    NNET_SCHEMA,
    SAFETYNET_SCHEMA,
    bundled_schema,
)
from castrainer.safetynet.validate import validate_safetynet_directory

app = typer.Typer(
    help="Validate SafetyNet directory structure and files",
    context_settings={"allow_interspersed_args": True},
)


@app.callback(invoke_without_command=True)
def validate(
    safetynet_dir: Annotated[
        Path,
        typer.Argument(help="Directory containing SafetyNet files"),
    ],
    manifest_schema: Annotated[
        Path | None,
        typer.Option(
            help="Path to manifest schema file. Defaults to the schema "
            "shipped with castrainer."
        ),
    ] = None,
    safetynet_schema: Annotated[
        Path | None,
        typer.Option(
            help="Path to SafetyNet schema file. Defaults to the schema "
            "shipped with castrainer."
        ),
    ] = None,
    nnet_schema: Annotated[
        Path | None,
        typer.Option(
            help=(
                'Path to nnet schema file, used to validate JSON-format ("jnet") '
                "networks. Defaults to the schema shipped with castrainer; "
                "networks stored as .pt/.pth/.onnx are unaffected either way."
            )
        ),
    ] = None,
) -> None:
    """Validate a SafetyNet (JSON files and network files).

    Args:
        safetynet_dir (Path): Directory containing SafetyNet files.
        manifest_schema (Path | None): Path to manifest schema file. If
            None, the schema bundled with castrainer is used.
        safetynet_schema (Path | None): Path to SafetyNet schema file.
            If None, the schema bundled with castrainer is used.
        nnet_schema (Path | None): Path to nnet schema file, used to
            validate JSON-format ("jnet") networks. If None, the schema
            bundled with castrainer is used.

    Raises:
        Exit: Exits with code 0 if validation passes, otherwise exits
            with code 1.
    """
    if not safetynet_dir.exists():
        logger.error(f"Directory not found: {safetynet_dir}")
        raise typer.Exit(code=1)

    logger.info(f"Validating SafetyNet directory: {safetynet_dir}")

    success = validate_safetynet_directory(
        safetynet_dir=safetynet_dir,
        manifest_schema_path=manifest_schema or bundled_schema(MANIFEST_SCHEMA),
        safetynet_schema_path=safetynet_schema or bundled_schema(SAFETYNET_SCHEMA),
        nnet_schema_path=nnet_schema or bundled_schema(NNET_SCHEMA),
    )

    if success:
        typer.echo("Validation passed successfully!")
        raise typer.Exit(code=0)
    typer.echo("Validation FAILED")
    raise typer.Exit(code=1)
