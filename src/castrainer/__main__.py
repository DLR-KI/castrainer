# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""The main entry point for the castrainer package."""

import logging
import warnings

from loguru import logger
from tqdm import TqdmExperimentalWarning

from castrainer.cli import app

# Silence deprecation warnings from third-party packages
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="isinstance.*treespec.*LeafSpec.*is deprecated",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="rich is experimental/alpha",
    category=TqdmExperimentalWarning,
)

# Configure the package-wide loguru sink
logger.remove()
logger.add(
    logging.StreamHandler(),
    format=(
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "{name}:{function}:{line} - {message}"
    ),
    level="INFO",
)


def main() -> None:
    """The main function for the castrainer package."""
    app()


if __name__ == "__main__":
    main()
