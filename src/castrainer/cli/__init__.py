# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""The CLI for the castrainer package."""

from castrainer.cli.cli import app
from castrainer.cli.utils import set_thread_limits

__all__ = ["app", "set_thread_limits"]
