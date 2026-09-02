# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for SafetyNet manifest/LUT/network validation rules."""

import json
from pathlib import Path
from typing import Any

from castrainer.safetynet.validate import (
    _validate_system_luts,
    _validate_system_networks,
    load_manifest,
    load_schema,
    validate_compatible_versioning,
    validate_condition_limits,
    validate_datatype_coherence,
    validate_ensured_responsibility,
    validate_files_available,
    validate_input_coherence,
    validate_input_coverage,
    validate_lut_correct_output,
    validate_lut_format,
    validate_lut_relayed_responsibility,
    validate_manifest,
    validate_network_correct_output,
    validate_network_format,
    validate_output_number,
    validate_safetynet_directory,
    validate_schema,
    validate_single_responsibility,
    validate_versioning,
    validate_wildcard_conditional,
)


def _base_manifest(**overrides: object) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "version": "1.0.0",
        "datatype": "float32",
        "numberOutputs": 2,
        "inputs": [
            {
                "id": "h",
                "ranges": [{"minimum": -10.0, "maximum": 10.0, "stride": 1.0}],
            },
        ],
        "networks": [],
        "luts": [],
    }
    manifest.update(overrides)
    return manifest


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)


def test_load_schema(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    _write_json(schema_path, {"type": "object"})
    assert load_schema(schema_path) == {"type": "object"}


def test_load_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, {"version": "1.0.0"})
    assert load_manifest(manifest_path) == {"version": "1.0.0"}


def test_validate_schema_valid() -> None:
    schema = {"type": "object", "required": ["version"]}
    assert validate_schema({"version": "1.0.0"}, schema) is True


def test_validate_schema_invalid() -> None:
    schema = {"type": "object", "required": ["version"]}
    assert validate_schema({}, schema) is False


def test_validate_files_available_all_present(tmp_path: Path) -> None:
    (tmp_path / "net.pt").touch()
    (tmp_path / "lut.json").touch()
    manifest = _base_manifest(
        networks=[{"file": "net.pt"}], luts=[{"file": "lut.json"}]
    )
    assert validate_files_available(manifest, tmp_path) is True


def test_validate_files_available_missing_network(tmp_path: Path) -> None:
    manifest = _base_manifest(networks=[{"file": "missing.pt"}])
    assert validate_files_available(manifest, tmp_path) is False


def test_validate_files_available_missing_lut(tmp_path: Path) -> None:
    manifest = _base_manifest(luts=[{"file": "missing.json"}])
    assert validate_files_available(manifest, tmp_path) is False


def test_validate_datatype_coherence_matching(tmp_path: Path) -> None:
    _write_json(tmp_path / "net.json", {"datatype": "float32"})
    manifest = _base_manifest(networks=[{"file": "net.json"}])
    assert validate_datatype_coherence(manifest, tmp_path) is True


def test_validate_datatype_coherence_mismatch(tmp_path: Path) -> None:
    _write_json(tmp_path / "net.json", {"datatype": "int8"})
    manifest = _base_manifest(networks=[{"file": "net.json"}])
    assert validate_datatype_coherence(manifest, tmp_path) is False


def test_validate_datatype_coherence_lut_mismatch(tmp_path: Path) -> None:
    _write_json(tmp_path / "lut.json", {"datatype": "int8"})
    manifest = _base_manifest(luts=[{"file": "lut.json"}])
    assert validate_datatype_coherence(manifest, tmp_path) is False


def test_validate_datatype_coherence_unreadable_file_is_lenient(
    tmp_path: Path,
) -> None:
    # A non-JSON file at a .json path can't be parsed; this is a
    # warning, not a hard failure.
    (tmp_path / "net.json").write_text("not json")
    manifest = _base_manifest(networks=[{"file": "net.json"}])
    assert validate_datatype_coherence(manifest, tmp_path) is True


def test_validate_datatype_coherence_ignores_non_json(tmp_path: Path) -> None:
    manifest = _base_manifest(networks=[{"file": "net.pt"}])
    assert validate_datatype_coherence(manifest, tmp_path) is True


def test_validate_input_coherence_known_input(tmp_path: Path) -> None:
    manifest = _base_manifest(
        networks=[{"file": "net.pt", "if": {"h": {"stride": 1.0}}}]
    )
    assert validate_input_coherence(manifest, tmp_path) is True


def test_validate_input_coherence_unknown_input(tmp_path: Path) -> None:
    manifest = _base_manifest(
        networks=[{"file": "net.pt", "if": {"unknown": {"stride": 1.0}}}]
    )
    assert validate_input_coherence(manifest, tmp_path) is False


def test_validate_input_coherence_stride_mismatch_is_warning_only(
    tmp_path: Path,
) -> None:
    manifest = _base_manifest(
        networks=[{"file": "net.pt", "if": {"h": {"stride": 99.0}}}]
    )
    # Stride mismatches are logged as warnings, not hard failures.
    assert validate_input_coherence(manifest, tmp_path) is True


def test_validate_input_coverage_full(tmp_path: Path) -> None:
    manifest = _base_manifest(
        networks=[{"file": "net.pt", "if": {"h": {"minimum": -10.0, "maximum": 10.0}}}]
    )
    assert validate_input_coverage(manifest, tmp_path) is True


def test_validate_input_coverage_no_conditions(tmp_path: Path) -> None:
    manifest = _base_manifest(networks=[{"file": "net.pt", "if": {}}])
    assert validate_input_coverage(manifest, tmp_path) is False


def test_validate_input_coverage_partial(tmp_path: Path) -> None:
    manifest = _base_manifest(
        networks=[{"file": "net.pt", "if": {"h": {"minimum": -5.0, "maximum": 5.0}}}]
    )
    assert validate_input_coverage(manifest, tmp_path) is False


def test_validate_output_number_matching(tmp_path: Path) -> None:
    _write_json(tmp_path / "lut.json", {"numberOutputs": 2})
    manifest = _base_manifest(luts=[{"file": "lut.json"}])
    assert validate_output_number(manifest, tmp_path) is True


def test_validate_output_number_mismatch(tmp_path: Path) -> None:
    _write_json(tmp_path / "lut.json", {"numberOutputs": 3})
    manifest = _base_manifest(luts=[{"file": "lut.json"}])
    assert validate_output_number(manifest, tmp_path) is False


def test_validate_condition_limits_within_bounds(tmp_path: Path) -> None:
    manifest = _base_manifest(
        networks=[{"file": "net.pt", "if": {"h": {"minimum": -5.0, "maximum": 5.0}}}]
    )
    assert validate_condition_limits(manifest, tmp_path) is True


def test_validate_condition_limits_exceeds_min(tmp_path: Path) -> None:
    manifest = _base_manifest(
        networks=[{"file": "net.pt", "if": {"h": {"minimum": -20.0, "maximum": 5.0}}}]
    )
    assert validate_condition_limits(manifest, tmp_path) is False


def test_validate_condition_limits_exceeds_max(tmp_path: Path) -> None:
    manifest = _base_manifest(
        networks=[{"file": "net.pt", "if": {"h": {"minimum": -5.0, "maximum": 20.0}}}]
    )
    assert validate_condition_limits(manifest, tmp_path) is False


def test_validate_condition_limits_unknown_input_skipped(tmp_path: Path) -> None:
    manifest = _base_manifest(
        networks=[
            {"file": "net.pt", "if": {"unknown": {"minimum": -999, "maximum": 999}}}
        ]
    )
    assert validate_condition_limits(manifest, tmp_path) is True


def test_validate_wildcard_conditional_valid() -> None:
    manifest = _base_manifest(networks=[{"file": "net.pt", "if": {"h": {}}}])
    assert validate_wildcard_conditional(manifest) is True


def test_validate_wildcard_conditional_unknown_input() -> None:
    manifest = _base_manifest(networks=[{"file": "net.pt", "if": {"unknown": {}}}])
    assert validate_wildcard_conditional(manifest) is False


def test_validate_ensured_responsibility_no_components(tmp_path: Path) -> None:
    manifest = _base_manifest()
    assert validate_ensured_responsibility(manifest, tmp_path) is False


def test_validate_ensured_responsibility_with_components(tmp_path: Path) -> None:
    manifest = _base_manifest(networks=[{"file": "net.pt", "if": {"h": {}}}])
    assert validate_ensured_responsibility(manifest, tmp_path) is True


def test_validate_single_responsibility_no_overlap() -> None:
    manifest = _base_manifest(
        networks=[
            {"file": "net1.pt", "if": {"h": {"minimum": 0, "maximum": 5}}},
            {"file": "net2.pt", "if": {"h": {"minimum": 5, "maximum": 10}}},
        ]
    )
    assert validate_single_responsibility(manifest) is True


def test_validate_single_responsibility_identical_conditions() -> None:
    manifest = _base_manifest(
        networks=[
            {"file": "net1.pt", "if": {"h": {"minimum": 0, "maximum": 5}}},
            {"file": "net2.pt", "if": {"h": {"minimum": 0, "maximum": 5}}},
        ]
    )
    assert validate_single_responsibility(manifest) is False


def test_validate_compatible_versioning_matching(tmp_path: Path) -> None:
    _write_json(tmp_path / "net.json", {"version": "1.0.0"})
    manifest = _base_manifest(networks=[{"file": "net.json"}])
    # Only warns on mismatch; always returns True currently.
    assert validate_compatible_versioning(manifest, tmp_path) is True


def test_validate_compatible_versioning_mismatch_still_valid(tmp_path: Path) -> None:
    _write_json(tmp_path / "net.json", {"version": "2.0.0"})
    manifest = _base_manifest(networks=[{"file": "net.json"}])
    assert validate_compatible_versioning(manifest, tmp_path) is True


def test_validate_compatible_versioning_lut(tmp_path: Path) -> None:
    _write_json(tmp_path / "lut.json", {"version": "1.0.0"})
    manifest = _base_manifest(luts=[{"file": "lut.json"}])
    assert validate_compatible_versioning(manifest, tmp_path) is True


def test_validate_compatible_versioning_unreadable(tmp_path: Path) -> None:
    (tmp_path / "net.json").write_text("not json")
    manifest = _base_manifest(networks=[{"file": "net.json"}])
    assert validate_compatible_versioning(manifest, tmp_path) is True


def test_validate_lut_correct_output_valid(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "lut.json",
        {
            "numberOutputs": 2,
            "datatype": "float32",
            "data": [{"inputs": [1.0], "outputs": [0.1, 0.2]}],
        },
    )
    manifest = _base_manifest(luts=[{"file": "lut.json"}])
    assert validate_lut_correct_output(manifest, tmp_path) is True


def test_validate_lut_correct_output_wrong_output_count(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "lut.json",
        {
            "numberOutputs": 2,
            "datatype": "float32",
            "data": [{"inputs": [1.0], "outputs": [0.1]}],
        },
    )
    manifest = _base_manifest(luts=[{"file": "lut.json"}])
    assert validate_lut_correct_output(manifest, tmp_path) is False


def test_validate_lut_correct_output_wrong_datatype(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "lut.json",
        {"numberOutputs": 2, "datatype": "int8", "data": []},
    )
    manifest = _base_manifest(luts=[{"file": "lut.json"}])
    assert validate_lut_correct_output(manifest, tmp_path) is False


def test_validate_lut_correct_output_unparseable(tmp_path: Path) -> None:
    (tmp_path / "lut.json").write_text("not json")
    manifest = _base_manifest(luts=[{"file": "lut.json"}])
    assert validate_lut_correct_output(manifest, tmp_path) is False


def test_validate_lut_correct_output_skips_non_json(tmp_path: Path) -> None:
    manifest = _base_manifest(luts=[{"file": "lut.bin"}])
    assert validate_lut_correct_output(manifest, tmp_path) is True


def test_validate_lut_relayed_responsibility_valid(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "lut.json",
        {"data": [{"inputs": [1.0], "outputs": [0.1]}]},
    )
    manifest = _base_manifest(luts=[{"file": "lut.json"}])
    assert validate_lut_relayed_responsibility(manifest, tmp_path) is True


def test_validate_lut_relayed_responsibility_missing_fields(tmp_path: Path) -> None:
    _write_json(tmp_path / "lut.json", {"data": [{"inputs": [1.0]}]})
    manifest = _base_manifest(luts=[{"file": "lut.json"}])
    assert validate_lut_relayed_responsibility(manifest, tmp_path) is False


def test_validate_lut_relayed_responsibility_unparseable(tmp_path: Path) -> None:
    (tmp_path / "lut.json").write_text("not json")
    manifest = _base_manifest(luts=[{"file": "lut.json"}])
    assert validate_lut_relayed_responsibility(manifest, tmp_path) is False


def test_validate_network_correct_output_valid(tmp_path: Path) -> None:
    _write_json(tmp_path / "net.json", {"numberOutputs": 2, "datatype": "float32"})
    manifest = _base_manifest(networks=[{"file": "net.json"}])
    assert validate_network_correct_output(manifest, tmp_path) is True


def test_validate_network_correct_output_wrong_output_count(tmp_path: Path) -> None:
    _write_json(tmp_path / "net.json", {"numberOutputs": 1, "datatype": "float32"})
    manifest = _base_manifest(networks=[{"file": "net.json"}])
    assert validate_network_correct_output(manifest, tmp_path) is False


def test_validate_network_correct_output_skips_pt_files(tmp_path: Path) -> None:
    manifest = _base_manifest(networks=[{"file": "net.pt"}])
    assert validate_network_correct_output(manifest, tmp_path) is True


def test_validate_network_correct_output_unparseable(tmp_path: Path) -> None:
    (tmp_path / "net.json").write_text("not json")
    manifest = _base_manifest(networks=[{"file": "net.json"}])
    assert validate_network_correct_output(manifest, tmp_path) is False


def test_validate_versioning_matching() -> None:
    manifest = {"version": "1.0.0"}
    schema = {"properties": {"version": {"const": "1.0.0"}}}
    assert validate_versioning(manifest, schema) is True


def test_validate_versioning_mismatch() -> None:
    manifest = {"version": "1.0.0"}
    schema = {"properties": {"version": {"const": "2.0.0"}}}
    assert validate_versioning(manifest, schema) is False


def test_validate_lut_format_valid() -> None:
    assert validate_lut_format(Path("lut.json")) is True


def test_validate_lut_format_invalid() -> None:
    assert validate_lut_format(Path("lut.csv")) is False


def test_validate_network_format_valid_variants() -> None:
    assert validate_network_format(Path("net.pt")) is True
    assert validate_network_format(Path("net.pth")) is True
    assert validate_network_format(Path("net.json")) is True
    assert validate_network_format(Path("net.onnx")) is True


def test_validate_network_format_invalid() -> None:
    assert validate_network_format(Path("net.exe")) is False


_SCHEMA = {
    "type": "object",
    "properties": {"version": {"const": "1.0.0"}},
    "required": ["version"],
}


def test_validate_manifest_all_pass(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    _write_json(schema_path, _SCHEMA)

    _write_json(tmp_path / "net.json", {"numberOutputs": 2, "datatype": "float32"})
    _write_json(
        tmp_path / "lut.json",
        {
            "numberOutputs": 2,
            "datatype": "float32",
            "data": [{"inputs": [1.0], "outputs": [0.1, 0.2]}],
        },
    )

    manifest = _base_manifest(
        networks=[
            {"file": "net.json", "if": {"h": {"minimum": -10.0, "maximum": 10.0}}}
        ],
        # The LUT covers the same subsystem but not an identical
        # condition set, so it doesn't trip G-008 (single
        # responsibility) against the network above.
        luts=[{"file": "lut.json", "if": {}}],
    )
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)

    assert validate_manifest(manifest_path, schema_path, tmp_path) is True


def test_validate_manifest_fails_on_missing_file(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    _write_json(schema_path, _SCHEMA)

    manifest = _base_manifest(networks=[{"file": "missing.pt"}])
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)

    assert validate_manifest(manifest_path, schema_path, tmp_path) is False


def test_validate_manifest_missing_manifest_file(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    _write_json(schema_path, _SCHEMA)
    assert (
        validate_manifest(tmp_path / "nonexistent.json", schema_path, tmp_path) is False
    )


def test_validate_safetynet_directory_all_valid(tmp_path: Path) -> None:
    manifest_schema_path = tmp_path / "manifest_schema.json"
    safetynet_schema_path = tmp_path / "safetynet_schema.json"
    _write_json(manifest_schema_path, _SCHEMA)
    _write_json(safetynet_schema_path, _SCHEMA)

    vcas_dir = tmp_path / "vcas"
    vcas_dir.mkdir()
    _write_json(vcas_dir / "net.json", {"numberOutputs": 2, "datatype": "float32"})
    manifest = _base_manifest(
        networks=[
            {"file": "net.json", "if": {"h": {"minimum": -10.0, "maximum": 10.0}}}
        ]
    )
    _write_json(vcas_dir / "vcas.json", manifest)

    # A sibling LUT/training-info style JSON file in the same directory
    # must NOT be picked up as if it were the system manifest.
    _write_json(vcas_dir / "vcas_01_training_info.json", {"not": "a manifest"})

    assert (
        validate_safetynet_directory(
            tmp_path, manifest_schema_path, safetynet_schema_path
        )
        is True
    )


def test_validate_safetynet_directory_invalid_component(tmp_path: Path) -> None:
    manifest_schema_path = tmp_path / "manifest_schema.json"
    safetynet_schema_path = tmp_path / "safetynet_schema.json"
    _write_json(manifest_schema_path, _SCHEMA)
    _write_json(safetynet_schema_path, _SCHEMA)

    hcas_dir = tmp_path / "hcas"
    hcas_dir.mkdir()
    manifest = _base_manifest(networks=[{"file": "missing.pt"}])
    _write_json(hcas_dir / "hcas.json", manifest)

    assert (
        validate_safetynet_directory(
            tmp_path, manifest_schema_path, safetynet_schema_path
        )
        is False
    )


def test_validate_safetynet_directory_empty(tmp_path: Path) -> None:
    manifest_schema_path = tmp_path / "manifest_schema.json"
    safetynet_schema_path = tmp_path / "safetynet_schema.json"
    _write_json(manifest_schema_path, _SCHEMA)
    _write_json(safetynet_schema_path, _SCHEMA)

    assert (
        validate_safetynet_directory(
            tmp_path, manifest_schema_path, safetynet_schema_path
        )
        is True
    )


def test_validate_safetynet_directory_missing_safetynet_schema(tmp_path: Path) -> None:
    manifest_schema_path = tmp_path / "manifest_schema.json"
    _write_json(manifest_schema_path, _SCHEMA)

    assert (
        validate_safetynet_directory(
            tmp_path, manifest_schema_path, tmp_path / "nonexistent_schema.json"
        )
        is False
    )


def _build_vcas_system_with_lut(tmp_path: Path, lut_data: dict[str, object]) -> Path:
    """Write a minimal but manifest-valid VCAS system with one LUT.

    Returns:
        Path: The created "vcas" system directory.
    """
    vcas_dir = tmp_path / "vcas"
    vcas_dir.mkdir()
    _write_json(vcas_dir / "net.json", {"numberOutputs": 2, "datatype": "float32"})
    _write_json(vcas_dir / "lut.json", lut_data)

    manifest = _base_manifest(
        networks=[
            {"file": "net.json", "if": {"h": {"minimum": -10.0, "maximum": 10.0}}}
        ],
        luts=[{"file": "lut.json", "if": {}}],
    )
    _write_json(vcas_dir / "vcas.json", manifest)
    return vcas_dir


def test_validate_safetynet_directory_referenced_lut_fails_schema(
    tmp_path: Path,
) -> None:
    """A LUT that satisfies the manifest's business rules but not the
    SafetyNet ("snet") schema (missing "version") must fail overall.
    """
    manifest_schema_path = tmp_path / "manifest_schema.json"
    safetynet_schema_path = tmp_path / "safetynet_schema.json"
    _write_json(manifest_schema_path, _SCHEMA)
    _write_json(safetynet_schema_path, _SCHEMA)

    _build_vcas_system_with_lut(
        tmp_path,
        {
            "numberOutputs": 2,
            "datatype": "float32",
            "data": [{"inputs": [1.0], "outputs": [0.1, 0.2]}],
        },
    )

    assert (
        validate_safetynet_directory(
            tmp_path, manifest_schema_path, safetynet_schema_path
        )
        is False
    )


def test_validate_safetynet_directory_referenced_lut_passes_schema(
    tmp_path: Path,
) -> None:
    manifest_schema_path = tmp_path / "manifest_schema.json"
    safetynet_schema_path = tmp_path / "safetynet_schema.json"
    _write_json(manifest_schema_path, _SCHEMA)
    _write_json(safetynet_schema_path, _SCHEMA)

    _build_vcas_system_with_lut(
        tmp_path,
        {
            "version": "1.0.0",
            "numberOutputs": 2,
            "datatype": "float32",
            "data": [{"inputs": [1.0], "outputs": [0.1, 0.2]}],
        },
    )

    assert (
        validate_safetynet_directory(
            tmp_path, manifest_schema_path, safetynet_schema_path
        )
        is True
    )


def test_validate_system_luts_pass(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "lut.json",
        {"version": "1.0.0", "numberOutputs": 1, "data": []},
    )
    manifest = _base_manifest(luts=[{"file": "lut.json"}])
    schema = {"type": "object", "required": ["version"]}
    assert _validate_system_luts(manifest, tmp_path, schema) is True


def test_validate_system_luts_schema_mismatch(tmp_path: Path) -> None:
    _write_json(tmp_path / "lut.json", {"numberOutputs": 1, "data": []})
    manifest = _base_manifest(luts=[{"file": "lut.json"}])
    schema = {"type": "object", "required": ["version"]}
    assert _validate_system_luts(manifest, tmp_path, schema) is False


def test_validate_system_luts_missing_file_skipped(tmp_path: Path) -> None:
    manifest = _base_manifest(luts=[{"file": "missing.json"}])
    schema = {"type": "object", "required": ["version"]}
    # Missing files are reported by G-001 elsewhere; this check must
    # not also fail because of them.
    assert _validate_system_luts(manifest, tmp_path, schema) is True


def test_validate_system_luts_ignores_non_json(tmp_path: Path) -> None:
    manifest = _base_manifest(luts=[{"file": "lut.bin"}])
    schema = {"type": "object", "required": ["version"]}
    assert _validate_system_luts(manifest, tmp_path, schema) is True


def test_validate_system_luts_unreadable(tmp_path: Path) -> None:
    (tmp_path / "lut.json").write_text("not json")
    manifest = _base_manifest(luts=[{"file": "lut.json"}])
    schema = {"type": "object", "required": ["version"]}
    assert _validate_system_luts(manifest, tmp_path, schema) is False


def test_validate_system_networks_pass(tmp_path: Path) -> None:
    _write_json(tmp_path / "net.json", {"version": "1.0.0", "numberOutputs": 1})
    manifest = _base_manifest(networks=[{"file": "net.json"}])
    schema = {"type": "object", "required": ["version"]}
    assert _validate_system_networks(manifest, tmp_path, schema) is True


def test_validate_system_networks_schema_mismatch(tmp_path: Path) -> None:
    _write_json(tmp_path / "net.json", {"numberOutputs": 1})
    manifest = _base_manifest(networks=[{"file": "net.json"}])
    schema = {"type": "object", "required": ["version"]}
    assert _validate_system_networks(manifest, tmp_path, schema) is False


def test_validate_system_networks_missing_file_skipped(tmp_path: Path) -> None:
    manifest = _base_manifest(networks=[{"file": "missing.json"}])
    schema = {"type": "object", "required": ["version"]}
    assert _validate_system_networks(manifest, tmp_path, schema) is True


def test_validate_system_networks_ignores_non_json(tmp_path: Path) -> None:
    # .pt/.pth/.onnx networks aren't JSON and can't be schema-checked
    # this way; N-001 (business-rule) checks cover them elsewhere.
    manifest = _base_manifest(networks=[{"file": "net.pt"}])
    schema = {"type": "object", "required": ["version"]}
    assert _validate_system_networks(manifest, tmp_path, schema) is True


def test_validate_system_networks_unreadable(tmp_path: Path) -> None:
    (tmp_path / "net.json").write_text("not json")
    manifest = _base_manifest(networks=[{"file": "net.json"}])
    schema = {"type": "object", "required": ["version"]}
    assert _validate_system_networks(manifest, tmp_path, schema) is False


def test_validate_safetynet_directory_nnet_schema_not_provided_skips_check(
    tmp_path: Path,
) -> None:
    """Without nnet_schema_path, JSON networks aren't schema-checked."""
    manifest_schema_path = tmp_path / "manifest_schema.json"
    safetynet_schema_path = tmp_path / "safetynet_schema.json"
    _write_json(manifest_schema_path, _SCHEMA)
    _write_json(safetynet_schema_path, _SCHEMA)

    vcas_dir = tmp_path / "vcas"
    vcas_dir.mkdir()
    # Missing "version", which would fail an nnet schema requiring it -
    # but since no nnet_schema_path is given, this must not matter.
    _write_json(vcas_dir / "net.json", {"numberOutputs": 2, "datatype": "float32"})
    manifest = _base_manifest(
        networks=[
            {"file": "net.json", "if": {"h": {"minimum": -10.0, "maximum": 10.0}}}
        ]
    )
    _write_json(vcas_dir / "vcas.json", manifest)

    assert (
        validate_safetynet_directory(
            tmp_path, manifest_schema_path, safetynet_schema_path
        )
        is True
    )


def test_validate_safetynet_directory_nnet_schema_fail(tmp_path: Path) -> None:
    manifest_schema_path = tmp_path / "manifest_schema.json"
    safetynet_schema_path = tmp_path / "safetynet_schema.json"
    nnet_schema_path = tmp_path / "nnet_schema.json"
    _write_json(manifest_schema_path, _SCHEMA)
    _write_json(safetynet_schema_path, _SCHEMA)
    _write_json(nnet_schema_path, _SCHEMA)

    vcas_dir = tmp_path / "vcas"
    vcas_dir.mkdir()
    # Passes N-001 (numberOutputs/datatype match) but is missing
    # "version", which the nnet schema requires.
    _write_json(vcas_dir / "net.json", {"numberOutputs": 2, "datatype": "float32"})
    manifest = _base_manifest(
        networks=[
            {"file": "net.json", "if": {"h": {"minimum": -10.0, "maximum": 10.0}}}
        ]
    )
    _write_json(vcas_dir / "vcas.json", manifest)

    assert (
        validate_safetynet_directory(
            tmp_path, manifest_schema_path, safetynet_schema_path, nnet_schema_path
        )
        is False
    )


def test_validate_safetynet_directory_nnet_schema_pass(tmp_path: Path) -> None:
    manifest_schema_path = tmp_path / "manifest_schema.json"
    safetynet_schema_path = tmp_path / "safetynet_schema.json"
    nnet_schema_path = tmp_path / "nnet_schema.json"
    _write_json(manifest_schema_path, _SCHEMA)
    _write_json(safetynet_schema_path, _SCHEMA)
    _write_json(nnet_schema_path, _SCHEMA)

    vcas_dir = tmp_path / "vcas"
    vcas_dir.mkdir()
    _write_json(
        vcas_dir / "net.json",
        {"version": "1.0.0", "numberOutputs": 2, "datatype": "float32"},
    )
    manifest = _base_manifest(
        networks=[
            {"file": "net.json", "if": {"h": {"minimum": -10.0, "maximum": 10.0}}}
        ]
    )
    _write_json(vcas_dir / "vcas.json", manifest)

    assert (
        validate_safetynet_directory(
            tmp_path, manifest_schema_path, safetynet_schema_path, nnet_schema_path
        )
        is True
    )


def test_validate_safetynet_directory_missing_nnet_schema_file(tmp_path: Path) -> None:
    manifest_schema_path = tmp_path / "manifest_schema.json"
    safetynet_schema_path = tmp_path / "safetynet_schema.json"
    _write_json(manifest_schema_path, _SCHEMA)
    _write_json(safetynet_schema_path, _SCHEMA)

    assert (
        validate_safetynet_directory(
            tmp_path,
            manifest_schema_path,
            safetynet_schema_path,
            tmp_path / "nonexistent_nnet_schema.json",
        )
        is False
    )
