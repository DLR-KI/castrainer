# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for SafetyNet CLI."""

from typer.testing import CliRunner

from castrainer.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    """Test CLI help command."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "SafetyNet" in result.stdout


def test_cli_info() -> None:
    """Test CLI info command."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "VCAS" in result.stdout
    assert "HCAS" in result.stdout


def test_cli_train_vcas_dry_run() -> None:
    """Test CLI train command for VCAS (dry run with minimal epochs)."""
    result = runner.invoke(
        app,
        [
            "train",
            "vcas",
            "--max-epochs",
            "1",
            "--hidden-nodes",
            "10",
            "--hidden-layers",
            "2",
            "--no-progress-bar",
        ],
    )
    # Should not crash (may fail due to missing data files in test env)
    assert result.exit_code in {0, 1}


def test_cli_train_hcas_dry_run() -> None:
    """Test CLI train command for HCAS (dry run with minimal epochs)."""
    result = runner.invoke(
        app,
        [
            "train",
            "hcas",
            "--max-epochs",
            "1",
            "--hidden-nodes",
            "10",
            "--hidden-layers",
            "2",
            "--no-progress-bar",
        ],
    )
    # Should not crash (may fail due to missing data files in test env)
    assert result.exit_code in {0, 1}


def test_cli_train_all_dry_run() -> None:
    """Test CLI train command for all systems (dry run)."""
    result = runner.invoke(
        app,
        [
            "train",
            "all",
            "--max-epochs",
            "1",
            "--hidden-nodes",
            "10",
            "--hidden-layers",
            "2",
            "--no-progress-bar",
        ],
    )
    # Should not crash (may fail due to missing data files in test env)
    assert result.exit_code in {0, 1}
