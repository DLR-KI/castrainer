# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Evaluate command: Convert JSON results to CSV."""

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any

import typer
from loguru import logger

app = typer.Typer(
    help="Convert JSON results to CSV files",
    context_settings={"allow_interspersed_args": True},
)


def _determine_case_type(filename: str) -> str:
    """Determine if file is HCAS or VCAS based on filename.

    Args:
        filename (str): The name of the file to analyze.

    Returns:
        str: A string indicating the case type: "HCAS", "VCAS", or
            "UNKNOWN".
    """
    filename_lower = filename.lower()
    if "hcas" in filename_lower or "horizontal" in filename_lower:
        return "HCAS"
    if "vcas" in filename_lower or "vertical" in filename_lower:
        return "VCAS"
    return "UNKNOWN"


def _load_json_file(json_file: Path, case_type: str) -> list[dict[str, Any]]:
    """Load one JSON result file, tagging entries with their case type.

    Args:
        json_file (Path): The JSON result file to load.
        case_type (str): The case type ("HCAS" or "VCAS") to tag
            entries with.

    Returns:
        list[dict[str, Any]]: The loaded entries, or an empty list if
            the file could not be read or parsed.
    """
    try:
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Error loading {json_file}: {e}")
        return []

    entries = data if isinstance(data, list) else [data]
    for entry in entries:
        entry["_case_type"] = case_type
    return entries


def _extract_activation_name(activation_str: str) -> str:
    """Extract simple activation name from a string.

    This function uses a regex to extract the base name of the
    activation function, ignoring any parameters or additional details.
    For example, "ReLU()" would return "ReLU".

    Args:
        activation_str (str): The raw activation string to process.

    Returns:
        str: A simplified activation name extracted from the input
            string.
    """
    match = re.match(r"(\w+)\(.*\)", activation_str)
    if match:
        return match.group(1)
    return activation_str


def _extract_layer_size(hidden_layers: list[int]) -> int:
    """Extract the layer size (assuming uniform layer sizes).

    Args:
        hidden_layers (list[int]): A list of integers representing the
        sizes of hidden layers.

    Returns:
        int: The size of the first hidden layer, or 0 if the list is
            empty.
    """
    if not hidden_layers:
        return 0
    return hidden_layers[0]


def _calculate_statistics(values: list[float]) -> dict[str, float]:
    """Calculate all statistics for a list of in_lut values.

    Args:
        values (list[float]): A list of in_lut values to analyze.

    Returns:
        dict[str, float]: A dictionary containing all calculated
            statistics, including min, max, mean, median, sigma,
            variance, and the values at +-1 to +-6 sigma.
    """
    if not values:
        return {}

    n = len(values)
    sorted_values = sorted(values)

    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    sigma = math.sqrt(variance)

    min_val = min(values)
    max_val = max(values)

    if n % 2 == 0:
        median = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    else:
        median = sorted_values[n // 2]

    return {
        "min": min_val,
        "max": max_val,
        "mean": mean,
        "median": median,
        "sigma": sigma,
        "var": variance,
        "+1sigma": mean + sigma,
        "-1sigma": mean - sigma,
        "+2sigma": mean + 2 * sigma,
        "-2sigma": mean - 2 * sigma,
        "+3sigma": mean + 3 * sigma,
        "-3sigma": mean - 3 * sigma,
        "+4sigma": mean + 4 * sigma,
        "-4sigma": mean - 4 * sigma,
        "+5sigma": mean + 5 * sigma,
        "-5sigma": mean - 5 * sigma,
        "+6sigma": mean + 6 * sigma,
        "-6sigma": mean - 6 * sigma,
    }


def _write_csv(
    output_path: Path,
    grouping_type: str,
    data: dict[int, list[float]],
) -> None:
    """Write statistics to a CSV file."""
    if not data:
        return

    group_col = "layer_size" if grouping_type == "layer_size" else "n_hidden_layers"

    sorted_keys = sorted(data.keys())

    rows = []
    for key in sorted_keys:
        values = data[key]
        stats = _calculate_statistics(values)
        if stats:
            row = {
                group_col: key,
                "in_lut_min": stats["min"],
                "in_lut_max": stats["max"],
                "in_lut_mean": stats["mean"],
                "in_lut_median": stats["median"],
                "in_lut_sigma": stats["sigma"],
                "in_lut_var": stats["var"],
                "+1sigma": stats["+1sigma"],
                "-1sigma": stats["-1sigma"],
                "+2sigma": stats["+2sigma"],
                "-2sigma": stats["-2sigma"],
                "+3sigma": stats["+3sigma"],
                "-3sigma": stats["-3sigma"],
                "+4sigma": stats["+4sigma"],
                "-4sigma": stats["-4sigma"],
                "+5sigma": stats["+5sigma"],
                "-5sigma": stats["-5sigma"],
                "+6sigma": stats["+6sigma"],
                "-6sigma": stats["-6sigma"],
            }
            rows.append(row)

    if not rows:
        return

    fieldnames = [
        group_col,
        "in_lut_min",
        "in_lut_max",
        "in_lut_mean",
        "in_lut_median",
        "in_lut_sigma",
        "in_lut_var",
        "+1sigma",
        "-1sigma",
        "+2sigma",
        "-2sigma",
        "+3sigma",
        "-3sigma",
        "+4sigma",
        "-4sigma",
        "+5sigma",
        "-5sigma",
        "+6sigma",
        "-6sigma",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"  Written: {output_path.name} ({len(rows)} rows)")


@app.callback(invoke_without_command=True)
def evaluate(  # ruff: ignore[complex-structure, too-many-branches]
    input_dir: Annotated[
        Path,
        typer.Argument(help="Directory containing JSON result files"),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(help="Output directory for CSV files"),
    ] = Path("data"),
) -> None:
    """Create CSV files from JSON result files.

    Args:
        input_dir (Path): The directory containing the JSON result files
            to process.
        output_dir (Path): The directory where the generated CSV files
            will be saved.
    """
    if not input_dir.exists():
        logger.error(f"Directory not found: {input_dir}")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all JSON files
    logger.info(f"Loading JSON files from {input_dir}...")
    all_results = []

    json_files = list(input_dir.glob("*.json"))
    json_files = [f for f in json_files if "status" not in f.stem]

    for json_file in json_files:
        case_type = _determine_case_type(json_file.name)
        if case_type == "UNKNOWN":
            continue

        logger.debug(f"Loading {json_file.name}...")
        all_results.extend(_load_json_file(json_file, case_type))

    if not all_results:
        logger.error("No valid data found!")
        return

    logger.info(f"Loaded {len(all_results)} results")

    # Process results and organize by case type, activation, one_hot,
    # and grouping
    by_layer_size: dict[str, dict[str, dict[bool, dict[int, list[float]]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    )
    by_n_layers: dict[str, dict[str, dict[bool, dict[int, list[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    for result in all_results:
        case_type = result.get("_case_type", "UNKNOWN")
        if case_type == "UNKNOWN":
            continue

        hidden_layers = result.get("hidden_layers", [])
        activation = _extract_activation_name(result.get("activation", "Unknown"))
        in_lut = result.get("in_lut")
        one_hot = result.get("one_hot", True)

        if in_lut is None or not hidden_layers:
            continue

        layer_size = _extract_layer_size(hidden_layers)
        n_hidden_layers = len(hidden_layers)

        by_layer_size[case_type][activation][one_hot][layer_size].append(in_lut)
        by_n_layers[case_type][activation][one_hot][n_hidden_layers].append(in_lut)

    # Write CSVs for layer size grouping
    logger.info("\nGenerating CSVs grouped by layer size:")
    for case_type in ["HCAS", "VCAS"]:
        if case_type not in by_layer_size:
            logger.info(f"  No {case_type} data found")
            continue
        for activation, one_hot_dict in by_layer_size[case_type].items():
            for one_hot, data in one_hot_dict.items():
                suffix = "-no-one-hot" if not one_hot else ""
                filename = f"data_{case_type}_{activation}_layer_size{suffix}.csv"
                _write_csv(output_dir / filename, "layer_size", data)

    # Write CSVs for n_hidden_layers grouping
    logger.info("\nGenerating CSVs grouped by number of hidden layers:")
    for case_type in ["HCAS", "VCAS"]:
        if case_type not in by_n_layers:
            continue
        for activation, one_hot_dict in by_n_layers[case_type].items():
            for one_hot, data in one_hot_dict.items():
                suffix = "-no-one-hot" if not one_hot else ""
                filename = f"data_{case_type}_{activation}_n_hidden_layers{suffix}.csv"
                _write_csv(output_dir / filename, "n_hidden_layers", data)

    logger.info(f"\nDone! CSV files saved to {output_dir}")
