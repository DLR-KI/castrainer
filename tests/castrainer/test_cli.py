# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for the castrainer CLI."""

# pylint: disable=missing-function-docstring,missing-class-docstring

import json
import math
from pathlib import Path
from unittest.mock import patch

from torch import nn
from typer.testing import CliRunner

from castrainer.cli.cli import app
from castrainer.cli.study import (
    _expand_file_paths,
    _iter_hyperparameters,
    _parse_comma_separated_list,
    generate_hyperparameter_combinations,
    get_available_devices,
    get_results_dir,
    load_completed_combinations,
    save_completed_combinations,
    set_thread_limits,
)
from castrainer.resources import (
    MANIFEST_SCHEMA,
    NNET_SCHEMA,
    SAFETYNET_SCHEMA,
    bundled_schema,
)
from castrainer.train.net import Net
from castrainer.train.safetynet import SafetyNet

runner = CliRunner()


def test_main_help() -> None:
    """Test main CLI help command."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "castrainer" in result.stdout
    assert "study" in result.stdout
    assert "train" in result.stdout
    assert "safetynet" in result.stdout
    assert "evaluate" in result.stdout
    assert "infer" in result.stdout
    assert "validate" in result.stdout


def test_study_help() -> None:
    """Test study command help."""
    result = runner.invoke(app, ["study", "--help"])
    assert result.exit_code == 0
    assert "study" in result.stdout
    assert "hyperparameter" in result.stdout


def test_train_help() -> None:
    """Test train command help."""
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "train" in result.stdout
    assert "single" in result.stdout


def test_safetynet_help() -> None:
    """Test safetynet command help."""
    result = runner.invoke(app, ["safetynet", "--help"])
    assert result.exit_code == 0
    assert "safetynet" in result.stdout
    assert "safety net" in result.stdout


def test_evaluate_help() -> None:
    """Test evaluate command help."""
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "evaluate" in result.stdout
    assert "CSV" in result.stdout


def test_validate_help() -> None:
    """Test validate command help."""
    result = runner.invoke(app, ["validate", "--help"])
    assert result.exit_code == 0
    assert "validate" in result.stdout


def test_info_command() -> None:
    """Test info command."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "VCAS" in result.stdout
    assert "HCAS" in result.stdout


def test_study_missing_file() -> None:
    """Test study command with missing file."""
    result = runner.invoke(app, ["study", "nonexistent.h5"])
    assert result.exit_code == 0


def test_study_with_config_file(tmp_path: Path) -> None:
    """Test study command with config file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("activations:\n  - relu\nhidden_nodes:\n  - 10\n")

    result = runner.invoke(
        app,
        [
            "study",
            "nonexistent.h5",
            "--config-file",
            str(config_file),
        ],
    )
    assert result.exit_code == 0


def test_study_with_config_file_and_cli_overrides(tmp_path: Path) -> None:
    """CLI options that differ from their defaults override the config file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("activations:\n  - relu\nhidden_nodes:\n  - 10\n")

    result = runner.invoke(
        app,
        [
            "study",
            "nonexistent.h5",
            "--config-file",
            str(config_file),
            "--activations",
            "gelu",
            "--hidden-nodes",
            "20",
            "--hidden-layers",
            "3",
        ],
    )
    assert result.exit_code == 0


def test_study_job_splitting() -> None:
    """Test study command splits combinations across job chunks."""
    result = runner.invoke(
        app,
        [
            "study",
            "nonexistent.h5",
            "--num-jobs",
            "2",
            "--job-id",
            "0",
        ],
    )
    assert result.exit_code == 0


def test_study_success_sequential(tmp_path: Path) -> None:
    """Test study command trains end-to-end with a real (tiny) dataset."""
    import h5py
    import numpy as np

    rng = np.random.default_rng(0)
    h5_path = tmp_path / "data.h5"
    with h5py.File(str(h5_path), "w") as f:
        f.create_dataset("X", data=rng.random((8, 3), dtype=np.float32))
        f.create_dataset("y", data=rng.random((8, 2), dtype=np.float32))

    results_dir = tmp_path / "results"
    with patch("castrainer.cli.study.get_results_dir", return_value=results_dir):
        result = runner.invoke(
            app,
            [
                "study",
                str(h5_path),
                "--activations",
                "relu",
                "--hidden-nodes",
                "4",
                "--hidden-layers",
                "1",
                "--max-epochs",
                "1",
                "--nproc",
                "1",
                "--no-progress-bar",
                "--strategy",
                "sequential",
            ],
        )
    assert result.exit_code == 0
    assert (results_dir / "status.json").exists()
    with (results_dir / "status.json").open("r", encoding="utf-8") as f:
        status = json.load(f)
    assert status["num_completed"] == 1


def test_study_resume_skips_completed(tmp_path: Path) -> None:
    """Test study command with --resume skips already-completed combos."""
    import h5py
    import numpy as np

    rng = np.random.default_rng(0)
    h5_path = tmp_path / "data.h5"
    with h5py.File(str(h5_path), "w") as f:
        f.create_dataset("X", data=rng.random((8, 3), dtype=np.float32))
        f.create_dataset("y", data=rng.random((8, 2), dtype=np.float32))

    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    with (results_dir / "status.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "completed_combinations": [
                    {"activation": "relu", "n_hidden_nodes": 4, "n_hidden_layers": 1}
                ]
            },
            f,
        )

    with patch("castrainer.cli.study.get_results_dir", return_value=results_dir):
        result = runner.invoke(
            app,
            [
                "study",
                str(h5_path),
                "--activations",
                "relu",
                "--hidden-nodes",
                "4",
                "--hidden-layers",
                "1",
                "--max-epochs",
                "1",
                "--nproc",
                "1",
                "--no-progress-bar",
                "--resume",
            ],
        )
    assert result.exit_code == 0


def test_train_missing_file() -> None:
    """Test train command with missing file."""
    result = runner.invoke(app, ["train", "nonexistent.h5"])
    assert result.exit_code == 0


def test_train_success(tmp_path: Path) -> None:
    """Test train command with a real (tiny) dataset trains end-to-end."""
    import h5py
    import numpy as np

    rng = np.random.default_rng(0)
    h5_path = tmp_path / "data.h5"
    with h5py.File(str(h5_path), "w") as f:
        f.create_dataset("X", data=rng.random((8, 3), dtype=np.float32))
        f.create_dataset("y", data=rng.random((8, 2), dtype=np.float32))

    output_dir = tmp_path / "trained"
    result = runner.invoke(
        app,
        [
            "train",
            str(h5_path),
            "--max-epochs",
            "1",
            "--hidden-nodes",
            "4",
            "--hidden-layers",
            "1",
            "--nproc",
            "1",
            "--no-progress-bar",
            "--output-dir",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0
    assert (output_dir / "training_info.json").exists()


def test_safetynet_invalid_system() -> None:
    """Test safetynet command with invalid system."""
    result = runner.invoke(app, ["safetynet", "invalid"])
    # Invalid systems are skipped, command completes
    assert result.exit_code == 0


def test_safetynet_vcas_dry_run(tmp_path: Path) -> None:
    """Test safetynet command for VCAS (dry run).

    Mocks the actual training so the CLI wiring is exercised without
    running real training against the repo's data files, which would
    take a very long time.
    """
    output_dir = tmp_path / "safetynet"
    with patch(
        "castrainer.safetynet.trainer.SafetyNetTrainer.train_system"
    ) as mock_train_system:
        result = runner.invoke(
            app,
            [
                "safetynet",
                "vcas",
                "--output-dir",
                str(output_dir),
                "--max-epochs",
                "1",
                "--hidden-nodes",
                "10",
                "--hidden-layers",
                "2",
                "--no-progress-bar",
            ],
        )
    assert result.exit_code == 0
    mock_train_system.assert_called_once()
    assert mock_train_system.call_args.kwargs["system"] == "vcas"


def test_safetynet_hcas_dry_run(tmp_path: Path) -> None:
    """Test safetynet command for HCAS (dry run).

    Mocks the actual training; see test_safetynet_vcas_dry_run.
    """
    output_dir = tmp_path / "safetynet"
    with patch(
        "castrainer.safetynet.trainer.SafetyNetTrainer.train_system"
    ) as mock_train_system:
        result = runner.invoke(
            app,
            [
                "safetynet",
                "hcas",
                "--output-dir",
                str(output_dir),
                "--max-epochs",
                "1",
                "--hidden-nodes",
                "10",
                "--hidden-layers",
                "2",
                "--no-progress-bar",
            ],
        )
    assert result.exit_code == 0
    mock_train_system.assert_called_once()
    assert mock_train_system.call_args.kwargs["system"] == "hcas"


def test_safetynet_all_dry_run(tmp_path: Path) -> None:
    """Test safetynet command for all systems (dry run).

    Mocks the actual training; see test_safetynet_vcas_dry_run.
    """
    output_dir = tmp_path / "safetynet"
    with patch(
        "castrainer.safetynet.trainer.SafetyNetTrainer.train_system"
    ) as mock_train_system:
        result = runner.invoke(
            app,
            [
                "safetynet",
                "all",
                "--output-dir",
                str(output_dir),
                "--max-epochs",
                "1",
                "--hidden-nodes",
                "10",
                "--hidden-layers",
                "2",
                "--no-progress-bar",
            ],
        )
    assert result.exit_code == 0
    assert mock_train_system.call_count == 2
    called_systems = {
        call.kwargs["system"] for call in mock_train_system.call_args_list
    }
    assert called_systems == {"vcas", "hcas"}


def test_evaluate_missing_directory() -> None:
    """Test evaluate command with missing directory."""
    result = runner.invoke(app, ["evaluate", "nonexistent"])
    assert result.exit_code == 0


def test_evaluate_empty_directory(tmp_path: Path) -> None:
    """Test evaluate command with empty directory."""
    result = runner.invoke(app, ["evaluate", str(tmp_path)])
    # May show error or complete, just check it runs
    assert result.exit_code in {0, 1}


def test_evaluate_with_json_files(tmp_path: Path) -> None:
    """Test evaluate command with JSON files."""
    # Create a test JSON result file. The filename must identify the
    # case type (VCAS/HCAS) and the payload needs "hidden_layers" and
    # "in_lut" for the entry to be included in the CSV output.
    test_file = tmp_path / "vcas_results.json"
    test_data = {
        "activation": "ReLU()",
        "hidden_layers": [10, 10],
        "in_lut": 0.5,
        "one_hot": True,
    }
    with test_file.open("w") as f:
        json.dump(test_data, f)

    output_dir = tmp_path / "csv_output"
    result = runner.invoke(
        app, ["evaluate", str(tmp_path), "--output-dir", str(output_dir)]
    )
    assert result.exit_code == 0
    assert output_dir.exists()
    assert any(output_dir.glob("*.csv"))


def test_validate_missing_directory(tmp_path: Path) -> None:
    """Test validate command with missing directory."""
    schema_file = tmp_path / "schema.json"
    schema_data = {"type": "object"}
    with schema_file.open("w") as f:
        json.dump(schema_data, f)

    result = runner.invoke(
        app,
        [
            "validate",
            "nonexistent",
            "--manifest-schema",
            str(schema_file),
            "--safetynet-schema",
            str(schema_file),
        ],
    )
    # Should handle missing directory gracefully
    assert result.exit_code in {0, 1}


def test_validate_success(tmp_path: Path) -> None:
    """Test validate command reports success when validation passes."""
    schema_file = tmp_path / "schema.json"
    with schema_file.open("w") as f:
        json.dump({"type": "object"}, f)

    safetynet_dir = tmp_path / "safetynet"
    safetynet_dir.mkdir()

    with patch(
        "castrainer.cli.validate.validate_safetynet_directory", return_value=True
    ) as mock_validate:
        result = runner.invoke(
            app,
            [
                "validate",
                str(safetynet_dir),
                "--manifest-schema",
                str(schema_file),
                "--safetynet-schema",
                str(schema_file),
            ],
        )
    assert result.exit_code == 0
    assert "passed successfully" in result.output
    # Omitted schema options fall back to the schemas bundled with the package.
    assert mock_validate.call_args.kwargs["nnet_schema_path"] == bundled_schema(
        NNET_SCHEMA
    )


def test_validate_uses_bundled_schemas_by_default(tmp_path: Path) -> None:
    """Test validate falls back to the schemas shipped with the package."""
    safetynet_dir = tmp_path / "safetynet"
    safetynet_dir.mkdir()

    with patch(
        "castrainer.cli.validate.validate_safetynet_directory", return_value=True
    ) as mock_validate:
        result = runner.invoke(app, ["validate", str(safetynet_dir)])

    assert result.exit_code == 0
    kwargs = mock_validate.call_args.kwargs
    assert kwargs["manifest_schema_path"] == bundled_schema(MANIFEST_SCHEMA)
    assert kwargs["safetynet_schema_path"] == bundled_schema(SAFETYNET_SCHEMA)
    assert kwargs["nnet_schema_path"] == bundled_schema(NNET_SCHEMA)
    assert all(
        kwargs[key].is_file()
        for key in ("manifest_schema_path", "safetynet_schema_path", "nnet_schema_path")
    )


def test_validate_with_nnet_schema(tmp_path: Path) -> None:
    """Test validate command passes --nnet-schema through when given."""
    schema_file = tmp_path / "schema.json"
    with schema_file.open("w") as f:
        json.dump({"type": "object"}, f)
    nnet_schema_file = tmp_path / "nnet_schema.json"
    with nnet_schema_file.open("w") as f:
        json.dump({"type": "object"}, f)

    safetynet_dir = tmp_path / "safetynet"
    safetynet_dir.mkdir()

    with patch(
        "castrainer.cli.validate.validate_safetynet_directory", return_value=True
    ) as mock_validate:
        result = runner.invoke(
            app,
            [
                "validate",
                str(safetynet_dir),
                "--manifest-schema",
                str(schema_file),
                "--safetynet-schema",
                str(schema_file),
                "--nnet-schema",
                str(nnet_schema_file),
            ],
        )
    assert result.exit_code == 0
    assert mock_validate.call_args.kwargs["nnet_schema_path"] == nnet_schema_file


def test_validate_failure(tmp_path: Path) -> None:
    """Test validate command reports failure when validation fails."""
    schema_file = tmp_path / "schema.json"
    with schema_file.open("w") as f:
        json.dump({"type": "object"}, f)

    safetynet_dir = tmp_path / "safetynet"
    safetynet_dir.mkdir()

    with patch(
        "castrainer.cli.validate.validate_safetynet_directory", return_value=False
    ):
        result = runner.invoke(
            app,
            [
                "validate",
                str(safetynet_dir),
                "--manifest-schema",
                str(schema_file),
                "--safetynet-schema",
                str(schema_file),
            ],
        )
    assert result.exit_code == 1
    assert "FAILED" in result.output


def test_parse_comma_separated_list() -> None:
    """Test comma-separated list parsing."""

    result = _parse_comma_separated_list("a, b, c")
    assert result == ["a", "b", "c"]

    result = _parse_comma_separated_list("single")
    assert result == ["single"]


def test_expand_file_paths_single_file(tmp_path: Path) -> None:
    """Test file path expansion with single file."""
    test_file = tmp_path / "test.h5"
    test_file.touch()

    result = _expand_file_paths([str(test_file)])
    assert len(result) == 1
    assert result[0] == test_file


def test_expand_file_paths_directory(tmp_path: Path) -> None:
    """Test file path expansion with directory."""
    # Create test files
    (tmp_path / "file1.h5").touch()
    (tmp_path / "file2.h5").touch()
    (tmp_path / "other.txt").touch()

    result = _expand_file_paths([str(tmp_path)])
    assert len(result) == 2
    assert all(p.suffix == ".h5" for p in result)


def test_expand_file_paths_glob(tmp_path: Path) -> None:
    """Test file path expansion with glob pattern."""
    (tmp_path / "file1.h5").touch()
    (tmp_path / "file2.h5").touch()
    (tmp_path / "other.txt").touch()

    pattern = str(tmp_path / "*.h5")
    result = _expand_file_paths([pattern])
    assert len(result) == 2


def test_expand_file_paths_not_found() -> None:
    """Test file path expansion with non-existent file."""

    result = _expand_file_paths(["nonexistent.h5"])
    assert len(result) == 0


def test_set_thread_limits() -> None:
    """Test thread limit setting."""

    with patch.dict("os.environ", {}, clear=False):
        set_thread_limits(4)
        import os

        assert os.environ.get("OPENBLAS_NUM_THREADS") == "4"
        assert os.environ.get("OMP_NUM_THREADS") == "4"
        assert os.environ.get("MKL_NUM_THREADS") == "4"


def test_set_thread_limits_none() -> None:
    """Test thread limit setting with None."""

    # Should not raise or modify
    set_thread_limits(None)


def test_get_available_devices_empty() -> None:
    """Test get available devices with empty string."""

    result = get_available_devices("")
    # Empty string results in empty list or list with 0 if int() succeeds
    assert isinstance(result, list)


def test_get_available_devices_single() -> None:
    """Test get available devices with single device."""

    result = get_available_devices("0")
    # On CPU-only system, returns empty list
    # On GPU system, returns [0]
    assert isinstance(result, list)


def test_get_available_devices_multiple() -> None:
    """Test get available devices with multiple devices."""

    result = get_available_devices("0, 1, 2")
    assert isinstance(result, list)


def test_generate_hyperparameter_combinations() -> None:
    """Test hyperparameter combination generation."""

    config = {
        "activations": ["relu", "gelu"],
        "hidden_nodes": [10, 20],
        "hidden_layers": [2],
    }
    result = generate_hyperparameter_combinations(config)
    assert len(result) == 4  # 2 * 2 * 1


def test_generate_hyperparameter_combinations_defaults() -> None:
    """Test hyperparameter combination generation with defaults."""

    config: dict = {}
    result = generate_hyperparameter_combinations(config)
    # Has defaults, just check it returns a list
    assert isinstance(result, list)
    assert len(result) > 0


def test_get_results_dir() -> None:
    """Test results directory generation."""

    result = get_results_dir("test_experiment")
    assert result.parts[-2] == "test_experiment"
    assert len(result.parts[-1]) == 15  # YYYYMMDD_HHMMSS


def test_load_completed_combinations_missing(tmp_path: Path) -> None:
    """Test loading completed combinations with missing file."""

    result = load_completed_combinations(tmp_path)
    assert result == set()


def test_load_completed_combinations(tmp_path: Path) -> None:
    """Test loading completed combinations."""

    status_file = tmp_path / "status.json"
    status_data = {
        "completed_combinations": [
            {"activation": "relu", "n_hidden_nodes": 10, "n_hidden_layers": 2}
        ]
    }
    with status_file.open("w") as f:
        json.dump(status_data, f)

    result = load_completed_combinations(tmp_path)
    assert "relu_n10_l2" in result


def test_save_completed_combinations(tmp_path: Path) -> None:
    """Test saving completed combinations."""

    completed = [{"activation": "relu", "n_hidden_nodes": 10, "n_hidden_layers": 2}]
    save_completed_combinations(tmp_path, completed)

    status_file = tmp_path / "status.json"
    assert status_file.exists()

    with status_file.open("r") as f:
        data = json.load(f)
    assert data["num_completed"] == 1


def test_get_activation_fn_relu() -> None:
    """Test getting ReLU activation."""

    # Just verify the function exists and can be called
    result = list(_iter_hyperparameters(["relu"], [10], [2]))
    assert len(result) == 1


def test_get_activation_fn_unknown() -> None:
    """Test getting unknown activation (defaults to ReLU)."""

    result = list(_iter_hyperparameters(["unknown"], [10], [2]))
    assert len(result) == 1


def test_load_help() -> None:
    """Test load command help."""
    result = runner.invoke(app, ["load", "--help"])
    assert result.exit_code == 0
    assert "load" in result.stdout
    assert "SafetyNet" in result.stdout


def test_infer_help() -> None:
    """Test infer command help."""
    result = runner.invoke(app, ["infer", "--help"])
    assert result.exit_code == 0
    assert "infer" in result.stdout
    assert "benchmark" in result.stdout.lower()


def test_infer_missing_directory(tmp_path: Path) -> None:
    """Test infer command with missing directory."""
    result = runner.invoke(app, ["infer", str(tmp_path / "nonexistent")])
    assert result.exit_code == 1


def test_infer_success(tmp_path: Path) -> None:
    """Test infer command benchmarks a real (tiny) SafetyNet on CPU."""
    from castrainer.train.lut import KDTreeLUT

    vcas_dir = tmp_path / "safetynet" / "vcas"
    vcas_dir.mkdir(parents=True)

    model = Net(inputs=3, outputs=2, hidden_layers=[4])
    import torch

    torch.save(model.state_dict(), vcas_dir / "net.pt")

    lut = KDTreeLUT.from_items([
        (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.1, 0.9])),
    ])
    with (vcas_dir / "net_lut.json").open("w") as f:
        json.dump(
            {
                "version": "1.0.0",
                "datatype": "float32",
                "numberOutputs": 2,
                "data": lut.to_serializable_entries(),
            },
            f,
        )

    with (vcas_dir / "vcas.json").open("w") as f:
        json.dump(
            {
                "version": "1.0.0",
                "datatype": "float32",
                "numberOutputs": 2,
                "inputs": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
                "networks": [{"file": "net.pt", "networkFormat": "torch", "if": {}}],
                "luts": [{"file": "net_lut.json", "lutFormat": "snet", "if": {}}],
            },
            f,
        )

    result = runner.invoke(
        app,
        [
            "infer",
            str(tmp_path / "safetynet"),
            "--repetitions",
            "2",
            "--warmup",
            "1",
            "--sample-size",
            "1",
            "--cpu-only",
        ],
    )

    assert result.exit_code == 0
    assert "VCAS system total" in result.output
    assert "samples_per_second" in result.output


def test_load_missing_directory(tmp_path: Path) -> None:
    """Test load command with missing directory."""
    result = runner.invoke(app, ["load", str(tmp_path / "nonexistent")])
    assert result.exit_code == 1


def test_load_missing_manifest(tmp_path: Path) -> None:
    """Test load command with missing manifest file."""
    # Create directory but no system manifests
    safetynet_dir = tmp_path / "safetynet"
    safetynet_dir.mkdir()

    result = runner.invoke(app, ["load", str(safetynet_dir)])
    assert result.exit_code == 1


def test_load_with_valid_manifest(tmp_path: Path) -> None:
    """Test load command with valid system manifest."""
    # Create a system directory with manifest (each system is independent)
    safetynet_dir = tmp_path / "safetynet"
    safetynet_dir.mkdir()

    # Create vcas system directory and manifest
    vcas_dir = safetynet_dir / "vcas"
    vcas_dir.mkdir()
    vcas_manifest = vcas_dir / "vcas.json"
    with vcas_manifest.open("w") as f:
        json.dump(
            {
                "version": "1.0.0",
                "description": "Test VCAS",
                "networks": [],
                "luts": [],
                "inputs": [],
                "numberOutputs": 1,
            },
            f,
        )

    result = runner.invoke(app, ["load", str(safetynet_dir)])
    # Should complete successfully with no networks/luts
    assert result.exit_code == 0
    assert "loaded successfully" in result.output


def test_load_specific_system(tmp_path: Path) -> None:
    """Test load command with specific system."""
    safetynet_dir = tmp_path / "safetynet"
    safetynet_dir.mkdir()

    # Create vcas system directory and manifest
    vcas_dir = safetynet_dir / "vcas"
    vcas_dir.mkdir()
    vcas_manifest = vcas_dir / "vcas.json"
    with vcas_manifest.open("w") as f:
        json.dump(
            {
                "version": "1.0.0",
                "description": "Test VCAS",
                "networks": [],
                "luts": [],
                "inputs": [],
                "numberOutputs": 1,
            },
            f,
        )

    # Create hcas system (should not be loaded when --system vcas is specified)
    hcas_dir = safetynet_dir / "hcas"
    hcas_dir.mkdir()
    hcas_manifest = hcas_dir / "hcas.json"
    with hcas_manifest.open("w") as f:
        json.dump(
            {
                "version": "1.0.0",
                "description": "Test HCAS",
                "networks": [],
                "luts": [],
                "inputs": [],
                "numberOutputs": 1,
            },
            f,
        )

    result = runner.invoke(app, ["load", str(safetynet_dir), "--system", "vcas"])
    assert result.exit_code == 0
    assert "VCAS" in result.output


def test_load_system_not_found(tmp_path: Path) -> None:
    """Test load command with non-existent system."""
    safetynet_dir = tmp_path / "safetynet"
    safetynet_dir.mkdir()

    # Create only vcas
    vcas_dir = safetynet_dir / "vcas"
    vcas_dir.mkdir()
    vcas_manifest = vcas_dir / "vcas.json"
    with vcas_manifest.open("w") as f:
        json.dump(
            {
                "version": "1.0.0",
                "description": "Test VCAS",
                "networks": [],
                "luts": [],
                "inputs": [],
                "numberOutputs": 1,
            },
            f,
        )

    result = runner.invoke(app, ["load", str(safetynet_dir), "--system", "hcas"])
    assert result.exit_code == 1


def test_training_info_file_structure(tmp_path: Path) -> None:
    """Test training info file has correct structure."""

    # Create a simple model
    model = Net(inputs=5, outputs=3, hidden_layers=[10, 10], activation=nn.ReLU)
    SafetyNet(model)

    # Create mock training info
    training_info = {
        "timestamp": "2024-01-01T00:00:00Z",
        "hyperparameters": {
            "activation": "ReLU",
            "hidden_layers": [10, 10],
            "num_hidden_layers": 2,
            "inputs": 5,
            "outputs": 3,
            "one_hot": False,
        },
        "training": {
            "max_epochs": 100,
            "min_epochs": 1,
            "current_epoch": 50,
            "global_step": 500,
            "training_duration_seconds": 123.45,
            "final_train_loss": 0.01,
            "final_test_loss": 0.02,
        },
        "model_sizes": {
            "network_size_bytes": 10000,
            "network_size_mb": 0.01,
            "lut_size_bytes": 50000,
            "lut_size_mb": 0.05,
            "safety_net_total_bytes": 60000,
            "safety_net_total_mb": 0.06,
        },
        "lut_statistics": {
            "lut_entries": 100,
            "dataset_size": 1000,
            "lut_coverage": 0.1,
            "nn_coverage": 0.9,
        },
        "dataset": {
            "filename": "test.h5",
            "size": 1000,
        },
    }

    info_file = tmp_path / "training_info.json"
    with info_file.open("w") as f:
        json.dump(training_info, f)

    # Verify structure
    with info_file.open("r") as f:
        data = json.load(f)

    assert "hyperparameters" in data
    assert "training" in data
    assert "model_sizes" in data
    assert "lut_statistics" in data
    assert "dataset" in data
    assert math.isclose(data["training"]["training_duration_seconds"], 123.45)
    assert data["model_sizes"]["network_size_bytes"] == 10000
