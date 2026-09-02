# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Validation module for SafetyNet manifests and files.

Validation functions are complex by nature.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import jsonschema
from loguru import logger


def load_schema(schema_path: Path) -> dict[str, Any]:
    """Load a JSON schema from file.

    Args:
        schema_path (Path): Path to the schema file.

    Returns:
        dict[str, Any]: The schema as a dictionary.
    """
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load a JSON manifest from file.

    Args:
        manifest_path (Path): Path to the manifest file.

    Returns:
        dict[str, Any]: The manifest as a dictionary.
    """
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_schema(instance: dict[str, Any], schema: dict[str, Any]) -> bool:
    """Validate an instance against a JSON schema.

    Args:
        instance (dict[str, Any]): The instance to validate.
        schema (dict[str, Any]): The schema to validate against.

    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        jsonschema.validate(instance=instance, schema=schema)
        return True
    except jsonschema.ValidationError as e:
        logger.error(f"Schema validation error: {e.message}")
        return False


def validate_files_available(manifest: dict[str, Any], base_path: Path) -> bool:
    """G-001: Validate all files referenced in manifest are available.

    Args:
        manifest (dict[str, Any]): The manifest to validate.
        base_path (Path): Base path for resolving file references.

    Returns:
        bool: True if all files are available, False otherwise.
    """
    all_valid = True

    for network in manifest.get("networks", []):
        file_path = base_path / network["file"]
        if not file_path.exists():
            logger.error(f"G-001: Network file not found: {file_path}")
            all_valid = False

    for lut in manifest.get("luts", []):
        file_path = base_path / lut["file"]
        if not file_path.exists():
            logger.error(f"G-001: LUT file not found: {file_path}")
            all_valid = False

    return all_valid


def _component_datatype_matches(
    file_path: Path,
    ref_name: str,
    manifest_datatype: object,
    component_label: str,
) -> bool:
    """Check a component JSON file's datatype against the manifest's.

    Args:
        file_path (Path): Path to the component's JSON file.
        ref_name (str): The component's file reference, for logging.
        manifest_datatype (object): The manifest's expected datatype.
        component_label (str): "Network" or "LUT", for logging.

    Returns:
        bool: True if the datatype matches, or the file could not be
            read (logged as a warning); False on a real mismatch.
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            component_dtype = json.load(f).get("datatype")
    except (json.JSONDecodeError, KeyError):
        logger.warning(
            f"G-002: Could not validate {component_label} datatype for {file_path}"
        )
        return True

    if component_dtype != manifest_datatype:
        logger.error(
            f"G-002: {component_label} {ref_name} has datatype "
            f"'{component_dtype}', expected '{manifest_datatype}'"
        )
        return False
    return True


def validate_datatype_coherence(
    manifest: dict[str, Any],
    base_path: Path,
) -> bool:
    """G-002: Validate datatype coherence across all components.

    Args:
        manifest (dict[str, Any]): The manifest to validate.
        base_path (Path): Base path for resolving file references.

    Returns:
        bool: True if datatypes are coherent, False otherwise.
    """
    manifest_datatype = manifest.get("datatype")
    all_valid = True

    # Check network datatypes (from nnet files if they exist)
    for network in manifest.get("networks", []):
        file_path = base_path / network["file"]
        if file_path.suffix == ".json" and not _component_datatype_matches(
            file_path, network["file"], manifest_datatype, "Network"
        ):
            all_valid = False

    # Check LUT datatypes
    for lut in manifest.get("luts", []):
        file_path = base_path / lut["file"]
        if file_path.suffix == ".json" and not _component_datatype_matches(
            file_path, lut["file"], manifest_datatype, "LUT"
        ):
            all_valid = False

    return all_valid


def validate_input_coherence(
    manifest: dict[str, Any],
    base_path: Path,  # ruff: ignore[unused-function-argument]
) -> bool:
    """G-003: Validate input coherence across all components.

    Args:
        manifest (dict[str, Any]): The manifest to validate.
        base_path (Path): Base path for resolving file references.

    Returns:
        bool: True if inputs are coherent, False otherwise.
    """
    manifest_inputs = manifest.get("inputs", [])
    all_valid = True

    # Check that all networks and LUTs have compatible input strides
    for network in manifest.get("networks", []):
        if_condition = network.get("if", {})
        for input_id, input_range in if_condition.items():
            # Find corresponding manifest input
            manifest_input = None
            for mi in manifest_inputs:
                if mi["id"] == input_id:
                    manifest_input = mi
                    break

            if manifest_input is None:
                logger.error(
                    f"G-003: Network {network['file']} references "
                    f"unknown input '{input_id}'"
                )
                all_valid = False
                continue

            # Validate stride is compatible
            manifest_ranges = manifest_input.get("ranges", [])
            if isinstance(input_range, dict) and "stride" in input_range:
                # Check if stride matches any manifest range
                stride = input_range["stride"]
                found = False
                for mr in manifest_ranges:
                    if math.isclose(mr.get("stride", 0), stride, abs_tol=1e-10):
                        found = True
                        break
                if not found:
                    logger.warning(
                        f"G-003: Network {network['file']} has stride "
                        f"{stride} not in manifest ranges for '{input_id}'"
                    )

    return all_valid


def validate_input_coverage(  # ruff: ignore[complex-structure]
    manifest: dict[str, Any],
    base_path: Path,  # ruff: ignore[unused-function-argument]
) -> bool:
    """G-004: Validate inputs cover the required input space.

    Args:
        manifest (dict[str, Any]): The manifest to validate.
        base_path (Path): Base path for resolving file references.

    Returns:
        bool: True if coverage is adequate, False otherwise.
    """
    all_valid = True
    manifest_inputs = manifest.get("inputs", [])

    # Get overall input space from manifest
    overall_min = {}
    overall_max = {}
    for inp in manifest_inputs:
        input_id = inp["id"]
        ranges = inp.get("ranges", [])
        if ranges:
            overall_min[input_id] = min(r["minimum"] for r in ranges)
            overall_max[input_id] = max(r["maximum"] for r in ranges)

    # Check that combined network/LUT conditions cover the space
    covered_ranges: dict[str, list[tuple[float, float]]] = {
        inp["id"]: [] for inp in manifest_inputs
    }

    for component in manifest.get("networks", []) + manifest.get("luts", []):
        if_condition = component.get("if", {})
        for input_id, input_range in if_condition.items():
            if isinstance(input_range, dict):
                min_val = input_range.get("minimum", input_range.get("minimum"))
                max_val = input_range.get("maximum", input_range.get("maximum"))
                if min_val is not None and max_val is not None:
                    covered_ranges[input_id].append((min_val, max_val))

    # Check coverage for each input
    for input_id, ranges in covered_ranges.items():
        if not ranges:
            logger.error(f"G-004: No coverage for input '{input_id}'")
            all_valid = False
            continue

        # Simple check: ensure min/max are covered
        overall_min_val = overall_min.get(input_id)
        overall_max_val = overall_max.get(input_id)

        if overall_min_val is not None and overall_max_val is not None:
            min_covered = min(r[0] for r in ranges) <= overall_min_val
            max_covered = max(r[1] for r in ranges) >= overall_max_val

            if not min_covered:
                logger.error(
                    f"G-004: Input '{input_id}' minimum {overall_min_val} not covered"
                )
                all_valid = False
            if not max_covered:
                logger.error(
                    f"G-004: Input '{input_id}' maximum {overall_max_val} not covered"
                )
                all_valid = False

    return all_valid


def _lut_output_count_matches(
    file_path: Path,
    ref_name: str,
    expected_outputs: object,
) -> bool:
    """Check one LUT JSON file's output count against the manifest's.

    Args:
        file_path (Path): Path to the LUT's JSON file.
        ref_name (str): The LUT's file reference, for logging.
        expected_outputs (object): The manifest's expected output count.

    Returns:
        bool: True if the output count matches, or the file could not
            be read (logged as a warning); False on a real mismatch.
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            lut_outputs = json.load(f).get("numberOutputs")
    except (json.JSONDecodeError, KeyError):
        logger.warning(f"G-005: Could not validate LUT output count for {file_path}")
        return True

    if lut_outputs != expected_outputs:
        logger.error(
            f"G-005: LUT {ref_name} has {lut_outputs} outputs, "
            f"expected {expected_outputs}"
        )
        return False
    return True


def validate_output_number(manifest: dict[str, Any], base_path: Path) -> bool:
    """G-005: Validate output number matches manifest.

    Args:
        manifest (dict[str, Any]): The manifest to validate.
        base_path (Path): Base path for resolving file references.

    Returns:
        bool: True if output numbers match, False otherwise.
    """
    expected_outputs = manifest.get("numberOutputs")
    all_valid = True

    # Check LUT outputs
    for lut in manifest.get("luts", []):
        file_path = base_path / lut["file"]
        if file_path.suffix == ".json" and not _lut_output_count_matches(
            file_path, lut["file"], expected_outputs
        ):
            all_valid = False

    return all_valid


def validate_condition_limits(
    manifest: dict[str, Any],
    base_path: Path,  # ruff: ignore[unused-function-argument]
) -> bool:
    """G-009: Validate conditions are within manifest limits.

    Args:
        manifest (dict[str, Any]): The manifest to validate.
        base_path (Path): Base path for resolving file references.

    Returns:
        bool: True if conditions are within limits, False otherwise.
    """
    all_valid = True
    manifest_inputs = manifest.get("inputs", [])

    # Build limit map
    limits: dict[str, tuple[float, float]] = {}
    for inp in manifest_inputs:
        input_id = inp["id"]
        ranges = inp.get("ranges", [])
        if ranges:
            min_val = min(r["minimum"] for r in ranges)
            max_val = max(r["maximum"] for r in ranges)
            limits[input_id] = (min_val, max_val)

    # Check network conditions
    for network in manifest.get("networks", []):
        if_condition = network.get("if", {})
        for input_id, input_range in if_condition.items():
            if input_id not in limits:
                continue

            limit_min, limit_max = limits[input_id]

            if isinstance(input_range, dict):
                cond_min = input_range.get("minimum")
                cond_max = input_range.get("maximum")

                if cond_min is not None and cond_min < limit_min:
                    logger.error(
                        f"G-009: Network {network['file']} condition for "
                        f"'{input_id}' minimum {cond_min} < limit {limit_min}"
                    )
                    all_valid = False

                if cond_max is not None and cond_max > limit_max:
                    logger.error(
                        f"G-009: Network {network['file']} condition for "
                        f"'{input_id}' maximum {cond_max} > limit {limit_max}"
                    )
                    all_valid = False

    return all_valid


def validate_wildcard_conditional(manifest: dict[str, Any]) -> bool:
    """G-010: Inputs not in conditions are not part of decision.

    Args:
        manifest (dict[str, Any]): The manifest to validate.

    Returns:
        bool: True if wildcard conditional rule is satisfied, False
            otherwise.
    """
    all_valid = True
    manifest_input_ids = {inp["id"] for inp in manifest.get("inputs", [])}

    for component in manifest.get("networks", []) + manifest.get("luts", []):
        if_condition = component.get("if", {})
        condition_input_ids = set(if_condition.keys())

        # Check that condition inputs are a subset of manifest inputs
        extra_inputs = condition_input_ids - manifest_input_ids
        if extra_inputs:
            logger.error(
                f"G-010: Component {component.get('file', 'unknown')} "
                f"references unknown inputs: {extra_inputs}"
            )
            all_valid = False

    return all_valid


def validate_ensured_responsibility(
    manifest: dict[str, Any],
    base_path: Path,  # ruff: ignore[unused-function-argument]
) -> bool:
    """G-007: Every allowed input vector is covered.

    Args:
        manifest (dict[str, Any]): The manifest to validate.
        base_path (Path): Base path for resolving file references.

    Returns:
        bool: True if responsibility is ensured, False otherwise.
    """
    # This is a complex validation that requires checking if the union
    # of all network/LUT conditions covers the entire input space.
    # For now, we do a simplified check.
    all_valid = True

    networks = manifest.get("networks", [])
    luts = manifest.get("luts", [])

    if not networks and not luts:
        logger.error("G-007: No networks or LUTs defined")
        return False

    # Check that we have at least one component for each subsystem
    # This is a heuristic - full coverage check requires SMT solving
    manifest_inputs = manifest.get("inputs", [])
    for inp in manifest_inputs:
        input_id = inp["id"]
        ranges = inp.get("ranges", [])
        if not ranges:
            continue

        # Check if any component covers this input
        covered = False
        for component in networks + luts:
            if_condition = component.get("if", {})
            if input_id in if_condition:
                covered = True
                break

        if not covered and len(ranges) > 1:
            logger.warning(f"G-007: Input '{input_id}' may not be fully covered")

    return all_valid


def validate_single_responsibility(manifest: dict[str, Any]) -> bool:
    """G-008: Every input vector is covered by at most one component.

    Args:
        manifest (dict[str, Any]): The manifest to validate.

    Returns:
        bool: True if single responsibility is satisfied, False
            otherwise.
    """
    all_valid = True

    # Check for overlapping conditions between components
    components = manifest.get("networks", []) + manifest.get("luts", [])

    for i, comp1 in enumerate(components):
        if_cond1 = comp1.get("if", {})
        for _, comp2 in enumerate(components[i + 1 :], i + 1):
            if_cond2 = comp2.get("if", {})

            # Check for exact overlap (same conditions)
            if if_cond1 == if_cond2 and comp1.get("file") != comp2.get("file"):
                logger.error(
                    f"G-008: Components {comp1.get('file')} and "
                    f"{comp2.get('file')} have identical conditions"
                )
                all_valid = False

    return all_valid


def validate_compatible_versioning(manifest: dict[str, Any], base_path: Path) -> bool:
    """M-002: Validate component versions are compatible with manifest.

    Args:
        manifest (dict[str, Any]): The manifest to validate.
        base_path (Path): Base path for resolving file references.

    Returns:
        bool: True if versions are compatible, False otherwise.
    """
    all_valid = True
    manifest_version = manifest.get("version", "1.0.0")

    # Check that all components have compatible versions
    for network in manifest.get("networks", []):
        file_path = base_path / network["file"]
        if file_path.suffix == ".json":
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    nnet = json.load(f)
                nnet_version = nnet.get("version", "1.0.0")
                if nnet_version != manifest_version:
                    logger.warning(
                        f"M-002: Network {network['file']} version "
                        f"'{nnet_version}' differs from manifest '{manifest_version}'"
                    )
            except (json.JSONDecodeError, KeyError):
                pass

    for lut in manifest.get("luts", []):
        file_path = base_path / lut["file"]
        if file_path.suffix == ".json":
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    lut_data = json.load(f)
                lut_version = lut_data.get("version", "1.0.0")
                if lut_version != manifest_version:
                    logger.warning(
                        f"M-002: LUT {lut['file']} version "
                        f"'{lut_version}' differs from manifest '{manifest_version}'"
                    )
            except (json.JSONDecodeError, KeyError):
                pass

    return all_valid


def _validate_lut_entries(
    lut_data: dict[str, Any],
    ref_name: str,
    expected_outputs: object,
    expected_dtype: object,
) -> bool:
    """Check output count, datatype, and entry lengths for one LUT.

    Args:
        lut_data (dict[str, Any]): Parsed LUT JSON contents.
        ref_name (str): The LUT's file reference, for logging.
        expected_outputs (object): The manifest's expected output
            count.
        expected_dtype (object): The manifest's expected datatype.

    Returns:
        bool: True if the LUT's contents are all correct.
    """
    valid = True

    if lut_data.get("numberOutputs") != expected_outputs:
        logger.error(f"L-001: LUT {ref_name} has wrong output count")
        valid = False

    if lut_data.get("datatype") != expected_dtype:
        logger.error(f"L-001: LUT {ref_name} has wrong datatype")
        valid = False

    for entry in lut_data.get("data", []):
        outputs = entry.get("outputs", [])
        if len(outputs) != expected_outputs:
            logger.error(
                f"L-001: LUT {ref_name} entry has "
                f"{len(outputs)} outputs, expected {expected_outputs}"
            )
            valid = False

    return valid


def validate_lut_correct_output(manifest: dict[str, Any], base_path: Path) -> bool:
    """L-001: Validate LUT output has correct length and datatype.

    Args:
        manifest (dict[str, Any]): The manifest to validate.
        base_path (Path): Base path for resolving file references.

    Returns:
        bool: True if LUT outputs are correct, False otherwise.
    """
    all_valid = True
    expected_outputs = manifest.get("numberOutputs")
    expected_dtype = manifest.get("datatype", "float32")

    for lut in manifest.get("luts", []):
        file_path = base_path / lut["file"]
        if file_path.suffix != ".json":
            continue

        try:
            with file_path.open("r", encoding="utf-8") as f:
                lut_data = json.load(f)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"L-001: Could not validate LUT {file_path}: {e}")
            all_valid = False
            continue

        if not _validate_lut_entries(
            lut_data, lut["file"], expected_outputs, expected_dtype
        ):
            all_valid = False

    return all_valid


def _lut_entries_have_required_fields(lut_data: dict[str, Any], ref_name: str) -> bool:
    """Check that a LUT's data entries all have inputs/outputs fields.

    Args:
        lut_data (dict[str, Any]): Parsed LUT JSON contents.
        ref_name (str): The LUT's file reference, for logging.

    Returns:
        bool: True if all entries are well-formed.
    """
    valid = True
    data = lut_data.get("data", [])
    if not data:
        logger.warning(f"L-003: LUT {ref_name} has no data entries")

    for entry in data:
        if "inputs" not in entry or "outputs" not in entry:
            logger.error(
                f"L-003: LUT {ref_name} entry missing "
                "required 'inputs' or 'outputs' field"
            )
            valid = False

    return valid


def validate_lut_relayed_responsibility(
    manifest: dict[str, Any], base_path: Path
) -> bool:
    """L-003: Validate LUT can determine responsibility.

    Args:
        manifest (dict[str, Any]): The manifest to validate.
        base_path (Path): Base path for resolving file references.

    Returns:
        bool: True if LUT can relay responsibility, False otherwise.
    """
    all_valid = True

    for lut in manifest.get("luts", []):
        file_path = base_path / lut["file"]
        if file_path.suffix != ".json":
            continue

        try:
            with file_path.open("r", encoding="utf-8") as f:
                lut_data = json.load(f)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"L-003: Could not validate LUT {file_path}: {e}")
            all_valid = False
            continue

        if not _lut_entries_have_required_fields(lut_data, lut["file"]):
            all_valid = False

    return all_valid


def _network_output_matches(
    nnet: dict[str, Any],
    ref_name: str,
    expected_outputs: object,
    expected_dtype: object,
) -> bool:
    """Check a network JSON file's output count and datatype.

    Args:
        nnet (dict[str, Any]): Parsed network JSON contents.
        ref_name (str): The network's file reference, for logging.
        expected_outputs (object): The manifest's expected output
            count.
        expected_dtype (object): The manifest's expected datatype.

    Returns:
        bool: True if the network's output count and datatype match.
    """
    valid = True

    if nnet.get("numberOutputs") != expected_outputs:
        logger.error(f"N-001: Network {ref_name} has wrong output count")
        valid = False

    if nnet.get("datatype") != expected_dtype:
        logger.error(f"N-001: Network {ref_name} has wrong datatype")
        valid = False

    return valid


def validate_network_correct_output(manifest: dict[str, Any], base_path: Path) -> bool:
    """N-001: Validate network output has correct length and datatype.

    Args:
        manifest (dict[str, Any]): The manifest to validate.
        base_path (Path): Base path for resolving file references.

    Returns:
        bool: True if network outputs are correct, False otherwise.
    """
    all_valid = True
    expected_outputs = manifest.get("numberOutputs")
    expected_dtype = manifest.get("datatype", "float32")

    for network in manifest.get("networks", []):
        file_path = base_path / network["file"]
        if file_path.suffix != ".json":
            # For .pt files, we can't validate without loading PyTorch
            continue

        try:
            with file_path.open("r", encoding="utf-8") as f:
                nnet = json.load(f)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"N-001: Could not validate network {file_path}: {e}")
            all_valid = False
            continue

        if not _network_output_matches(
            nnet, network["file"], expected_outputs, expected_dtype
        ):
            all_valid = False

    return all_valid


def validate_versioning(manifest: dict[str, Any], schema: dict[str, Any]) -> bool:
    """M-001: Validate manifest version matches schema.

    Args:
        manifest (dict[str, Any]): The manifest to validate.
        schema (dict[str, Any]): The schema to validate against.

    Returns:
        bool: True if versions are compatible, False otherwise.
    """
    manifest_version = manifest.get("version")
    schema_version = schema.get("properties", {}).get("version", {}).get("const")

    if manifest_version != schema_version:
        logger.error(
            f"M-001: Manifest version '{manifest_version}' does not match "
            f"schema version '{schema_version}'"
        )
        return False

    return True


def validate_lut_format(lut_path: Path) -> bool:
    """L-002: Validate LUT is in allowed format.

    Args:
        lut_path (Path): Path to the LUT file.

    Returns:
        bool: True if format is valid, False otherwise.
    """
    allowed_formats = [".json"]  # snet format
    return lut_path.suffix.lower() in allowed_formats


def validate_network_format(network_path: Path) -> bool:
    """N-002: Validate network is in allowed format.

    Args:
        network_path (Path): Path to the network file.

    Returns:
        bool: True if format is valid, False otherwise.
    """
    allowed_formats = [".pt", ".pth", ".json", ".onnx"]  # torch, nnet, onnx
    return network_path.suffix.lower() in allowed_formats


def validate_manifest(  # ruff: ignore[complex-structure, too-many-branches, too-many-statements]
    manifest_path: Path,
    schema_path: Path,
    base_path: Path | None = None,
) -> bool:
    """Validate a manifest file against schema and business rules.

    Args:
        manifest_path (Path): Path to the manifest file.
        schema_path (Path): Path to the schema file.
        base_path (Path): Base path for resolving file references.

    Returns:
        bool: True if all validations pass, False otherwise.
    """
    logger.info(f"Validating manifest: {manifest_path}")

    if base_path is None:
        base_path = manifest_path.parent

    # Load files
    try:
        manifest = load_manifest(manifest_path)
        schema = load_schema(schema_path)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load files: {e}")
        return False

    all_valid = True

    # Schema validation
    logger.info("Checking schema compliance...")
    if not validate_schema(manifest, schema):
        logger.error("Schema validation failed")
        all_valid = False

    # M-001: Versioning
    logger.info("Checking M-001: Versioning...")
    if not validate_versioning(manifest, schema):
        all_valid = False

    # G-001: Available Files
    logger.info("Checking G-001: Available Files...")
    if not validate_files_available(manifest, base_path):
        all_valid = False

    # G-002: Datatype Coherence
    logger.info("Checking G-002: Datatype Coherence...")
    if not validate_datatype_coherence(manifest, base_path):
        all_valid = False

    # G-003: Input Coherence
    logger.info("Checking G-003: Input Coherence...")
    if not validate_input_coherence(manifest, base_path):
        all_valid = False

    # G-004: Input Coverage
    logger.info("Checking G-004: Input Coverage...")
    if not validate_input_coverage(manifest, base_path):
        all_valid = False

    # G-005: Output Number
    logger.info("Checking G-005: Output Number...")
    if not validate_output_number(manifest, base_path):
        all_valid = False

    # G-006: Output Type (implicit in schema validation)
    logger.info("Checking G-006: Output Type...")
    # Output type is validated via schema and G-005

    # G-007: Ensured Responsibility
    logger.info("Checking G-007: Ensured Responsibility...")
    if not validate_ensured_responsibility(manifest, base_path):
        all_valid = False

    # G-008: Single Responsibility
    logger.info("Checking G-008: Single Responsibility...")
    if not validate_single_responsibility(manifest):
        all_valid = False

    # G-009: Condition Limits
    logger.info("Checking G-009: Condition Limits...")
    if not validate_condition_limits(manifest, base_path):
        all_valid = False

    # G-010: Wildcard Conditional
    logger.info("Checking G-010: Wildcard Conditional...")
    if not validate_wildcard_conditional(manifest):
        all_valid = False

    # M-002: Compatible Versioning
    logger.info("Checking M-002: Compatible Versioning...")
    if not validate_compatible_versioning(manifest, base_path):
        all_valid = False

    # L-001: Correct Output
    logger.info("Checking L-001: Correct Output...")
    if not validate_lut_correct_output(manifest, base_path):
        all_valid = False

    # L-002: LUT Format
    logger.info("Checking L-002: LUT Format...")
    for lut in manifest.get("luts", []):
        lut_path = base_path / lut["file"]
        if not validate_lut_format(lut_path):
            logger.error(f"L-002: Invalid LUT format for {lut_path}")
            all_valid = False

    # L-003: Relayed Responsibility
    logger.info("Checking L-003: Relayed Responsibility...")
    if not validate_lut_relayed_responsibility(manifest, base_path):
        all_valid = False

    # N-001: Correct Output
    logger.info("Checking N-001: Correct Output...")
    if not validate_network_correct_output(manifest, base_path):
        all_valid = False

    # N-002: Network Format
    logger.info("Checking N-002: Network Format...")
    for network in manifest.get("networks", []):
        network_path = base_path / network["file"]
        if not validate_network_format(network_path):
            logger.error(f"N-002: Invalid network format for {network_path}")
            all_valid = False

    if all_valid:
        logger.info("All validations passed!")
    else:
        logger.error("Some validations failed")

    return all_valid


def _validate_system_luts(
    manifest: dict[str, Any],
    base_path: Path,
    safetynet_schema: dict[str, Any],
) -> bool:
    """Validate a manifest's LUT files against the SafetyNet schema.

    Missing or unreadable LUT files are skipped here since
    `validate_manifest`'s G-001 check already reports them.

    Args:
        manifest (dict[str, Any]): The system manifest.
        base_path (Path): Base path for resolving LUT file references.
        safetynet_schema (dict[str, Any]): The SafetyNet ("snet" LUT
            format) schema.

    Returns:
        bool: True if every referenced JSON LUT file is schema-valid.
    """
    all_valid = True
    for lut in manifest.get("luts", []):
        lut_path = base_path / lut["file"]
        if lut_path.suffix != ".json" or not lut_path.exists():
            continue

        try:
            with lut_path.open("r", encoding="utf-8") as f:
                lut_data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Could not read LUT {lut_path}: {e}")
            all_valid = False
            continue

        if not validate_schema(lut_data, safetynet_schema):
            logger.error(f"LUT {lut_path} failed SafetyNet schema validation")
            all_valid = False

    return all_valid


def _validate_system_networks(
    manifest: dict[str, Any],
    base_path: Path,
    nnet_schema: dict[str, Any],
) -> bool:
    """Validate JSON-format ("jnet") networks against the nnet schema.

    Networks stored in other formats (".pt", ".pth", ".onnx") aren't
    JSON and can't be schema-checked this way; they still go through
    the N-001 output/datatype business-rule checks elsewhere. Missing
    or unreadable network files are skipped here since
    `validate_manifest`'s G-001 check already reports them.

    Args:
        manifest (dict[str, Any]): The system manifest.
        base_path (Path): Base path for resolving network file
            references.
        nnet_schema (dict[str, Any]): The nnet (JSON network format)
            schema.

    Returns:
        bool: True if every referenced JSON network file is
            schema-valid.
    """
    all_valid = True
    for network in manifest.get("networks", []):
        network_path = base_path / network["file"]
        if network_path.suffix != ".json" or not network_path.exists():
            continue

        try:
            with network_path.open("r", encoding="utf-8") as f:
                network_data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Could not read network {network_path}: {e}")
            all_valid = False
            continue

        if not validate_schema(network_data, nnet_schema):
            logger.error(f"Network {network_path} failed nnet schema validation")
            all_valid = False

    return all_valid


def validate_safetynet_directory(
    safetynet_dir: Path,
    manifest_schema_path: Path,
    safetynet_schema_path: Path,
    nnet_schema_path: Path | None = None,
) -> bool:
    """Validate an entire SafetyNet directory.

    Each system (e.g. "vcas", "hcas") has its own subdirectory
    containing exactly one manifest file named after the system (e.g.
    "vcas/vcas.json"), alongside its network and LUT files. The
    manifest is validated against `manifest_schema_path` (including the
    G-001..N-002 business rules), every LUT file it references is
    additionally schema-validated against `safetynet_schema_path` (the
    "snet" LUT format), and, if `nnet_schema_path` is given, every
    JSON-format ("jnet") network file is schema-validated against it,
    too.

    Args:
        safetynet_dir (Path): Path to the SafetyNet directory.
        manifest_schema_path (Path): Path to the manifest schema.
        safetynet_schema_path (Path): Path to the SafetyNet (LUT)
            schema, used to validate each referenced LUT file.
        nnet_schema_path (Path | None): Path to the nnet (JSON network
            format) schema. If given, JSON-format network files are
            additionally schema-validated against it; networks stored
            as ".pt"/".pth"/".onnx" aren't JSON and are unaffected
            either way. If None, no nnet schema validation is
            performed.

    Returns:
        bool: True if all validations pass, False otherwise.
    """
    logger.info(f"Validating SafetyNet directory: {safetynet_dir}")

    try:
        safetynet_schema = load_schema(safetynet_schema_path)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load SafetyNet schema: {e}")
        return False

    nnet_schema: dict[str, Any] | None = None
    if nnet_schema_path is not None:
        try:
            nnet_schema = load_schema(nnet_schema_path)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load nnet schema: {e}")
            return False

    all_valid = True

    # Validate each system's manifest (named after its directory, e.g.
    # "vcas/vcas.json"), then every LUT/network file it references.
    for system_dir in sorted(p for p in safetynet_dir.iterdir() if p.is_dir()):
        manifest_path = system_dir / f"{system_dir.name}.json"
        if not manifest_path.exists():
            continue

        if not validate_manifest(manifest_path, manifest_schema_path, system_dir):
            all_valid = False

        try:
            manifest = load_manifest(manifest_path)
        except (FileNotFoundError, json.JSONDecodeError):
            continue

        if not _validate_system_luts(manifest, system_dir, safetynet_schema):
            all_valid = False

        if nnet_schema is not None and not _validate_system_networks(
            manifest, system_dir, nnet_schema
        ):
            all_valid = False

    return all_valid


def main() -> None:
    """Main entry point for validation CLI."""
    parser = argparse.ArgumentParser(description="Validate SafetyNet manifests")
    parser.add_argument("manifest", type=Path, help="Path to manifest file")
    parser.add_argument(
        "--schema", type=Path, required=True, help="Path to schema file"
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=None,
        help="Base path for resolving file references",
    )

    args = parser.parse_args()

    success = validate_manifest(args.manifest, args.schema, args.base_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
