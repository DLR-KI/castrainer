# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Info command: Show information about available systems."""

import typer

from castrainer.safetynet.config import HCAS_CONFIG, VCAS_CONFIG

app = typer.Typer(help="Show information about available systems and configurations")


@app.callback(invoke_without_command=True)
def info() -> None:
    """Show information about available systems and configurations."""
    typer.echo(
        "SafetyNet Training System\n"
        "Available systems:\n"
        "VCAS (Vertical Collision Avoidance System):\n"
        f"  - Subsystems: {VCAS_CONFIG['num_subsystems']}\n"
        f"  - Inputs: {VCAS_CONFIG['num_inputs']}\n"
        f"  - Outputs: {VCAS_CONFIG['num_outputs']}\n"
        f"  - Description: {VCAS_CONFIG['description']}\n"
        "HCAS (Horizontal Collision Avoidance System):\n"
        f"  - Subsystems: {HCAS_CONFIG['num_subsystems']}\n"
        f"  - Inputs: {HCAS_CONFIG['num_inputs']}\n"
        f"  - Outputs: {HCAS_CONFIG['num_outputs']}\n"
        f"  - Description: {HCAS_CONFIG['description']}\n"
    )
