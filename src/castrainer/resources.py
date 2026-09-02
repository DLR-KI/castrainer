# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Access to data files shipped inside the castrainer package."""

from importlib import resources
from pathlib import Path

MANIFEST_SCHEMA = "manifest.schema.json"
NNET_SCHEMA = "nnet.schema.json"
SAFETYNET_SCHEMA = "safetynet.schema.json"


def bundled_schema(name: str) -> Path:
    """Get the path to a JSON schema shipped with the package.

    The schemas live in ``src/castrainer/schemas/`` and are installed
    as package data, so they are available both from a source checkout
    and from an installed wheel.

    Args:
        name (str): File name of the schema, e.g. ``MANIFEST_SCHEMA``.

    Returns:
        Path: Path to the bundled schema file.
    """
    return Path(str(resources.files("castrainer").joinpath("schemas", name)))
