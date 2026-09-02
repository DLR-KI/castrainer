# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""The CLI for the castrainer package.

This module imports and combines all sub-command Typer apps into a
single CLI.
"""

import typer

from castrainer.cli.evaluate import app as evaluate_app
from castrainer.cli.infer import app as infer_app
from castrainer.cli.info import app as info_app
from castrainer.cli.load import app as load_app
from castrainer.cli.safetynet import app as safetynet_app
from castrainer.cli.study import app as study_app
from castrainer.cli.train import app as train_app
from castrainer.cli.validate import app as validate_app

app = typer.Typer(
    name="castrainer",
    help="Train neural networks for HCAS and VCAS systems",
)

# Include all sub-apps
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(info_app, name="info")
app.add_typer(infer_app, name="infer")
app.add_typer(load_app, name="load")
app.add_typer(safetynet_app, name="safetynet")
app.add_typer(study_app, name="study")
app.add_typer(train_app, name="train")
app.add_typer(validate_app, name="validate")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
