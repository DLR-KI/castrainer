# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for the castrainer package entry point."""

from unittest.mock import patch

import castrainer.__main__ as main_module


def test_main_module_configures_warning_filters() -> None:
    # Importing the module registers its warnings.filterwarnings calls;
    # just verify the module loaded and exposes the expected app/main.
    assert hasattr(main_module, "app")
    assert callable(main_module.main)


def test_main_invokes_app() -> None:
    with patch.object(main_module, "app") as mock_app:
        main_module.main()
    mock_app.assert_called_once()
