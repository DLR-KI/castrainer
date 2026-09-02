# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""SafetyNet loading helpers."""

import json
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import nn

from castrainer.train.lut import KDTreeLUT
from castrainer.train.net import Net
from castrainer.train.safetynet import SafetyNet


@dataclass(slots=True)
class LoadedSafetyNet:
    """A loaded SafetyNet bundle."""

    name: str
    model: Net
    safetynet: SafetyNet
    network_file: Path
    lut_file: Path | None
    manifest: dict[str, Any]
    lut_serialized_size_bytes: int | None = None
    lut_kdtree_size_bytes: int | None = None


def _infer_model_architecture(
    state: dict[str, torch.Tensor],
    default_inputs: int,
    default_outputs: int,
) -> tuple[int, int, list[int]]:
    """Infer Net dimensions from a state dict.

    Args:
        state (dict[str, torch.Tensor]): The model's state dict.
        default_inputs (int): Fallback input count.
        default_outputs (int): Fallback output count.

    Returns:
        tuple[int, int, list[int]]: A (inputs, outputs, hidden_layers)
            tuple, falling back to the given defaults and an empty
            hidden-layer list if the state dict has no recognizable
            encoder weights.
    """
    encoder_weights = []
    for key, value in state.items():
        if key.startswith("encoder.") and key.endswith(".weight"):
            try:
                layer_index = int(key.split(".")[1])
            except (IndexError, ValueError):
                continue
            encoder_weights.append((layer_index, value))

    if not encoder_weights:
        return default_inputs, default_outputs, []

    encoder_weights.sort(key=operator.itemgetter(0))
    shapes = [weight.shape for _, weight in encoder_weights]
    inferred_inputs = int(shapes[0][1])
    inferred_outputs = int(shapes[-1][0])
    hidden_layers = [int(shape[0]) for shape in shapes[:-1]]
    return inferred_inputs, inferred_outputs, hidden_layers


def load_safetynet_directory(
    safetynet_dir: Path,
    system: str | None = None,
) -> dict[str, list[LoadedSafetyNet]]:
    """Load SafetyNet bundles from a directory.

    Args:
        safetynet_dir (Path): Directory containing per-system
            SafetyNet subdirectories.
        system (str | None): If given, load only this system ("vcas"
            or "hcas"). Otherwise, all available systems are loaded.

    Returns:
        dict[str, list[LoadedSafetyNet]]: A mapping from system name
            to its loaded bundles.

    Raises:
        FileNotFoundError: If `safetynet_dir` doesn't exist, or no
            valid system manifests are found within it.
        ValueError: If `system` is given but not available.
    """
    if not safetynet_dir.exists():
        raise FileNotFoundError(f"Directory not found: {safetynet_dir}")

    available_systems = _discover_available_systems(safetynet_dir)
    if not available_systems:
        raise FileNotFoundError(
            f"No valid system manifests found in {safetynet_dir}. "
            "Expected subdirectories: vcas, hcas"
        )

    if system is not None:
        if system not in available_systems:
            raise ValueError(
                f"System '{system}' not found. Available systems: "
                f"{', '.join(available_systems)}"
            )
        systems_to_load = [system]
    else:
        systems_to_load = available_systems

    loaded: dict[str, list[LoadedSafetyNet]] = {}
    for sys_name in systems_to_load:
        bundles = _load_system(safetynet_dir, sys_name)
        loaded[sys_name] = bundles

    return loaded


def _discover_available_systems(safetynet_dir: Path) -> list[str]:
    """Find systems under `safetynet_dir` with a valid manifest.

    Args:
        safetynet_dir (Path): Directory containing per-system
            SafetyNet subdirectories.

    Returns:
        list[str]: The available system names ("vcas" and/or "hcas").
    """
    valid_systems = {"vcas", "hcas"}
    available_systems: list[str] = []

    for subdir in safetynet_dir.iterdir():
        if subdir.is_dir() and subdir.name in valid_systems:
            system_manifest = subdir / f"{subdir.name}.json"
            if system_manifest.exists():
                available_systems.append(subdir.name)

    return available_systems


def _file_size_or_none(path: Path) -> int | None:
    """Return a file's on-disk size, or None if it can't be stat'd.

    Args:
        path (Path): The file to measure.

    Returns:
        int | None: The file's size in bytes, or None on failure.
    """
    try:
        return int(path.stat().st_size)
    except OSError:
        return None


def _kdtree_size_or_none(lut: KDTreeLUT) -> int | None:
    """Return a LUT's in-memory k-d tree size, or None on failure.

    This is a best-effort diagnostic measurement, not required for
    correct loading, so any failure to introspect the underlying
    SciPy/Torch objects is swallowed rather than propagated.

    Args:
        lut (KDTreeLUT): The LUT to measure.

    Returns:
        int | None: The estimated size in bytes, or None on failure.
    """
    try:
        return int(lut.size_bytes())
    except Exception:  # ruff: ignore[blind-except]
        return None


def _load_network_lut(
    safetynet_dir: Path,
    sys_name: str,
    index: int,
    lut_infos: list[dict[str, Any]],
    safetynet: SafetyNet,
) -> tuple[Path | None, int | None, int | None]:
    """Load and attach a network's LUT, if the manifest references one.

    Args:
        safetynet_dir (Path): Directory containing per-system
            SafetyNet subdirectories.
        sys_name (str): The system name ("vcas" or "hcas").
        index (int): The network's index within the manifest.
        lut_infos (list[dict[str, Any]]): The manifest's LUT entries.
        safetynet (SafetyNet): The SafetyNet to attach the loaded LUT
            to.

    Returns:
        tuple[Path | None, int | None, int | None]: A (lut_file,
            serialized_size_bytes, kdtree_size_bytes) tuple; all None
            if there is no LUT for this network.
    """
    if index >= len(lut_infos):
        return None, None, None

    lut_file = safetynet_dir / sys_name / lut_infos[index]["file"]
    if not lut_file.exists():
        logger.warning(f"LUT file not found: {lut_file}")
        return lut_file, None, None

    with lut_file.open("r", encoding="utf-8") as f:
        lut_data = json.load(f)

    serialized_size = _file_size_or_none(lut_file)
    safetynet.lut = KDTreeLUT.from_serialized_entries(lut_data.get("data", []))
    kdtree_size = _kdtree_size_or_none(safetynet.lut)

    if serialized_size is not None or kdtree_size is not None:
        logger.info(
            f"Loaded LUT {lut_file.name}: serialized={serialized_size} bytes, "
            f"kdtree={kdtree_size} bytes"
        )

    return lut_file, serialized_size, kdtree_size


@dataclass(slots=True)
class _SystemLoadContext:
    """Shared, per-system context for loading network bundles."""

    safetynet_dir: Path
    sys_name: str
    sys_manifest: dict[str, Any]
    default_inputs: int
    default_outputs: int
    lut_infos: list[dict[str, Any]]


def _load_network_bundle(
    ctx: _SystemLoadContext,
    index: int,
    net_info: dict[str, Any],
) -> LoadedSafetyNet | None:
    """Load one network (and its LUT, if present) into a bundle.

    Args:
        ctx (_SystemLoadContext): Shared context for this system.
        index (int): The network's index within the manifest.
        net_info (dict[str, Any]): The manifest's entry for this
            network.

    Returns:
        LoadedSafetyNet | None: The loaded bundle, or None if the
            network file is missing.
    """
    net_file = ctx.safetynet_dir / ctx.sys_name / net_info["file"]
    if not net_file.exists():
        logger.warning(f"Network file not found: {net_file}")
        return None

    state = torch.load(net_file, map_location="cpu", weights_only=True)

    # Infer architecture from state dict to match saved model shape
    inferred_inputs, inferred_outputs, hidden_layers_inferred = (
        _infer_model_architecture(state, ctx.default_inputs, ctx.default_outputs)
    )

    model = Net(
        inputs=inferred_inputs,
        outputs=inferred_outputs,
        hidden_layers=hidden_layers_inferred,
        activation=nn.ReLU,
        one_hot=False,
    )
    model.load_state_dict(state)

    safetynet = SafetyNet(model)
    lut_file, serialized_size, kdtree_size = _load_network_lut(
        ctx.safetynet_dir, ctx.sys_name, index, ctx.lut_infos, safetynet
    )

    return LoadedSafetyNet(
        name=Path(net_info["file"]).stem,
        model=model,
        safetynet=safetynet,
        network_file=net_file,
        lut_file=lut_file,
        lut_serialized_size_bytes=serialized_size,
        lut_kdtree_size_bytes=kdtree_size,
        manifest=ctx.sys_manifest,
    )


def _load_system(safetynet_dir: Path, sys_name: str) -> list[LoadedSafetyNet]:
    """Load all network/LUT bundles for one system.

    Args:
        safetynet_dir (Path): Directory containing per-system
            SafetyNet subdirectories.
        sys_name (str): The system name ("vcas" or "hcas").

    Returns:
        list[LoadedSafetyNet]: The loaded bundles, or an empty list if
            the system manifest is missing.
    """
    system_manifest_path = safetynet_dir / sys_name / f"{sys_name}.json"
    if not system_manifest_path.exists():
        logger.warning(f"System manifest not found: {system_manifest_path}")
        return []

    with system_manifest_path.open("r", encoding="utf-8") as f:
        sys_manifest = json.load(f)

    ctx = _SystemLoadContext(
        safetynet_dir=safetynet_dir,
        sys_name=sys_name,
        sys_manifest=sys_manifest,
        default_inputs=len(sys_manifest.get("inputs", [])),
        default_outputs=int(sys_manifest.get("numberOutputs", 1)),
        lut_infos=sys_manifest.get("luts", []),
    )

    networks: list[LoadedSafetyNet] = []
    for index, net_info in enumerate(sys_manifest.get("networks", [])):
        bundle = _load_network_bundle(ctx, index, net_info)
        if bundle is not None:
            networks.append(bundle)

    return networks
