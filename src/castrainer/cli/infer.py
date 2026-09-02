# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Infer command: benchmark SafetyNet inference time."""

import math
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Annotated

import numpy as np
import torch
import typer
from loguru import logger
from tqdm.rich import tqdm

from castrainer.safetynet.io import LoadedSafetyNet, load_safetynet_directory

app = typer.Typer(
    help="Benchmark SafetyNet inference time",
    context_settings={"allow_interspersed_args": True},
)

# Above this many LUT points, cap sample_size/repetitions to keep the
# benchmark practical.
_LARGE_LUT_THRESHOLD = 200_000


@dataclass(slots=True)
class BenchmarkStats:
    """Per-batch inference statistics."""

    mean_seconds_per_sample: float
    std_seconds_per_sample: float
    min_seconds_per_sample: float
    max_seconds_per_sample: float
    samples_per_second: float
    total_samples: int


@dataclass(slots=True)
class _BundleBenchmark:
    stats: BenchmarkStats
    repetitions: int
    batch_size: int


def _build_benchmark_batch(
    bundle: LoadedSafetyNet,
    sample_size: int,
) -> tuple[torch.Tensor, int, int]:
    """Build a mixed hit/miss batch for inference benchmarking.

    Args:
        bundle (LoadedSafetyNet): The SafetyNet bundle to sample LUT
            points from.
        sample_size (int): Number of LUT hits to sample.

    Returns:
        tuple[torch.Tensor, int, int]: A (batch, hit_count,
            miss_count) tuple.
    """
    safetynet = bundle.safetynet
    points = safetynet.lut.points
    if len(points) == 0:
        return torch.zeros((1, safetynet.model.inputs), dtype=torch.float32), 0, 1

    hit_count = min(sample_size, len(points))
    rng = np.random.default_rng(0)
    indices = np.arange(len(points))
    rng.shuffle(indices)
    selected_points = points[indices[:hit_count]]

    hit_batch = torch.tensor(selected_points, dtype=torch.float32)
    miss_batch = hit_batch.clone()
    miss_batch[:, 0] += 1.0

    batch = torch.cat([hit_batch, miss_batch], dim=0)
    return batch, hit_count, hit_count


def _benchmark_predict(
    bundle: LoadedSafetyNet,
    batch: torch.Tensor,
    repetitions: int,
    warmup: int,
    device: torch.device,
) -> _BundleBenchmark:
    """Benchmark repeated prediction calls.

    Args:
        bundle (LoadedSafetyNet): The SafetyNet bundle to benchmark.
        batch (torch.Tensor): The input batch to repeatedly predict
            on.
        repetitions (int): Number of timed prediction calls.
        warmup (int): Number of untimed warmup calls before timing
            starts.
        device (torch.device): Device to run the benchmark on.

    Returns:
        _BundleBenchmark: The resulting timing statistics and run
            configuration.
    """
    safetynet = bundle.safetynet
    safetynet.model.to(device)
    batch = batch.to(device)

    with torch.inference_mode():
        for _ in range(warmup):
            safetynet.predict(batch)

    per_sample_seconds: list[float] = []
    for _ in range(repetitions):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            safetynet.predict(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        per_sample_seconds.append(elapsed / batch.shape[0])

    total_samples = repetitions * batch.shape[0]
    mean_seconds_per_sample = fmean(per_sample_seconds) if per_sample_seconds else 0.0
    std_seconds_per_sample = (
        stdev(per_sample_seconds) if len(per_sample_seconds) > 1 else 0.0
    )
    min_seconds_per_sample = min(per_sample_seconds) if per_sample_seconds else 0.0
    max_seconds_per_sample = max(per_sample_seconds) if per_sample_seconds else 0.0
    samples_per_second = (
        1.0 / mean_seconds_per_sample if mean_seconds_per_sample > 0 else float("inf")
    )

    stats = BenchmarkStats(
        mean_seconds_per_sample=mean_seconds_per_sample,
        std_seconds_per_sample=std_seconds_per_sample,
        min_seconds_per_sample=min_seconds_per_sample,
        max_seconds_per_sample=max_seconds_per_sample,
        samples_per_second=samples_per_second,
        total_samples=total_samples,
    )

    return _BundleBenchmark(
        stats=stats, repetitions=repetitions, batch_size=int(batch.shape[0])
    )


@dataclass(slots=True)
class _AggregationTotals:
    """Running totals accumulated across a system's bundle results."""

    timed_samples: int = 0
    time_sum: float = 0.0  # sum of elapsed seconds across all timed samples
    sumsq: float = 0.0  # sum of squared per-sample times, across all
    # sample-observations
    min_time: float = float("inf")
    max_time: float = 0.0

    def add_bundle(self, result: _BundleBenchmark) -> None:
        """Fold one bundle's benchmark result into the totals.

        Args:
            result (_BundleBenchmark): The bundle's timing result to
                fold in.
        """
        stats = result.stats
        k = result.repetitions
        b = result.batch_size

        self.timed_samples += k * b
        self.time_sum += stats.mean_seconds_per_sample * k * b
        # sum of squares of per-sample times (per repetition), scaled
        # by batch size
        sumsq_per_rep = ((k - 1) * (stats.std_seconds_per_sample**2)) + (
            k * (stats.mean_seconds_per_sample**2)
        )
        self.sumsq += b * sumsq_per_rep

        self.min_time = min(self.min_time, stats.min_seconds_per_sample)
        self.max_time = max(self.max_time, stats.max_seconds_per_sample)


def _aggregate_system(
    results: list[tuple[LoadedSafetyNet, _BundleBenchmark]],
) -> BenchmarkStats:
    """Aggregate per-bundle benchmark results into system-level stats.

    Args:
        results (list[tuple[LoadedSafetyNet, _BundleBenchmark]]):
            Per-bundle benchmark results to aggregate.

    Returns:
        BenchmarkStats: The combined stats across all given bundles.
    """
    totals = _AggregationTotals()
    for _bundle, result in results:
        totals.add_bundle(result)

    if totals.timed_samples == 0:
        return BenchmarkStats(
            mean_seconds_per_sample=0.0,
            std_seconds_per_sample=0.0,
            min_seconds_per_sample=0.0,
            max_seconds_per_sample=0.0,
            samples_per_second=float("inf"),
            total_samples=0,
        )

    mean = totals.time_sum / totals.timed_samples
    mean_sq = totals.sumsq / totals.timed_samples
    var = max(0.0, mean_sq - (mean * mean))
    std = math.sqrt(var)
    samples_per_second = 1.0 / mean if mean > 0 else float("inf")

    min_time = totals.min_time if totals.min_time != float("inf") else 0.0
    return BenchmarkStats(
        mean_seconds_per_sample=mean,
        std_seconds_per_sample=std,
        min_seconds_per_sample=min_time,
        max_seconds_per_sample=totals.max_time,
        samples_per_second=samples_per_second,
        total_samples=totals.timed_samples,
    )


@app.callback(invoke_without_command=True)
def infer(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    safetynet_dir: Annotated[
        Path,
        typer.Argument(help="Directory containing SafetyNet files"),
    ],
    system: Annotated[
        str | None,
        typer.Option(
            help=(
                "System to benchmark (vcas or hcas). If None, discovers all "
                "available systems."
            )
        ),
    ] = None,
    repetitions: Annotated[
        int,
        typer.Option(help="Number of benchmark repetitions"),
    ] = 100,
    warmup: Annotated[
        int,
        typer.Option(help="Number of warmup runs"),
    ] = 10,
    sample_size: Annotated[
        int,
        typer.Option(min=1, help="Number of LUT hits to sample per bundle"),
    ] = 5000,
    cpu_only: Annotated[  # ruff: ignore[boolean-default-value-positional-argument]
        bool,
        typer.Option(help="Force CPU-only benchmarking"),
    ] = False,
) -> None:
    """Report inference time for loaded SafetyNet models.

    Args:
        safetynet_dir (Path): Directory containing SafetyNet files.
        system (str | None): System to benchmark (vcas or hcas). If
            None, discovers all available systems.
        repetitions (int): Number of benchmark repetitions.
        warmup (int): Number of warmup runs.
        sample_size (int): Number of LUT hits to sample per bundle.
        cpu_only (bool): Force CPU-only benchmarking.

    Raises:
        Exit: If the SafetyNet directory or system can't be loaded.
    """
    # suppress per-LUT loader info lines (they are noisy); re-enable
    # later
    logger.disable("castrainer.safetynet.io")
    try:
        loaded = load_safetynet_directory(safetynet_dir, system)
    except (FileNotFoundError, ValueError) as exc:
        logger.enable("castrainer.safetynet.io")
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc
    finally:
        logger.enable("castrainer.safetynet.io")

    device = torch.device(
        "cpu" if cpu_only or not torch.cuda.is_available() else "cuda"
    )

    typer.echo("SafetyNet inference benchmark")
    typer.echo(f"  device={device}")
    typer.echo(f"  sample_size={sample_size} hits per bundle")
    # show progress over systems and bundles using tqdm (rich if
    # available)
    systems = list(loaded.keys())
    with tqdm(systems, desc=f"Systems ({len(systems)})", unit="system") as systems_bar:
        for sys_name in systems_bar:
            bundles = loaded[sys_name]
            systems_bar.set_postfix_str(sys_name.upper())
            results: list[tuple[LoadedSafetyNet, _BundleBenchmark]] = []
            with tqdm(
                bundles,
                desc=f"{sys_name.upper()} bundles ({len(bundles)})",
                unit="bundle",
                leave=False,
            ) as bundle_bar:
                for bundle in bundle_bar:
                    # For very large LUTs, avoid expensive operations
                    # and auto-cap
                    points_len = len(bundle.safetynet.lut.points)

                    local_sample_size = sample_size
                    local_repetitions = repetitions
                    if points_len > _LARGE_LUT_THRESHOLD:
                        # large LUT: reduce work to keep benchmark
                        # practical
                        local_sample_size = min(sample_size, 1000)
                        local_repetitions = min(repetitions, 10)
                        typer.echo(
                            f"  Large LUT ({points_len} entries)---capping "
                            f"sample_size to {local_sample_size} and "
                            f"repetitions to {local_repetitions} for speed."
                        )

                    batch, _, _ = _build_benchmark_batch(
                        bundle,
                        local_sample_size,
                    )
                    result = _benchmark_predict(
                        bundle,
                        batch,
                        local_repetitions,
                        warmup,
                        device,
                    )
                    results.append((bundle, result))

            # aggregate per-system and print a single summary (one
            # key per line)
            aggregated = _aggregate_system(results)

            # compute totals for models and LUTs
            total_model_bytes = 0
            total_kdtree_bytes = 0
            for bundle in bundles:
                total_model_bytes += sum(
                    p.numel() * p.element_size() for p in bundle.model.parameters()
                )
                total_kdtree_bytes += int(bundle.lut_kdtree_size_bytes or 0)

            # print one key:value per line, no indentation
            typer.echo(f"{sys_name.upper()} system total")
            typer.echo(
                f"mean_seconds_per_sample: {aggregated.mean_seconds_per_sample:.6e}"
            )
            typer.echo(
                f"std_seconds_per_sample: {aggregated.std_seconds_per_sample:.6e}"
            )
            typer.echo(
                f"min_seconds_per_sample: {aggregated.min_seconds_per_sample:.6e}"
            )
            typer.echo(
                f"max_seconds_per_sample: {aggregated.max_seconds_per_sample:.6e}"
            )
            typer.echo(f"samples_per_second: {aggregated.samples_per_second:.2f}")
            typer.echo(f"total_inference_runs: {aggregated.total_samples}")
            typer.echo(f"total_model_bytes: {total_model_bytes}")
            typer.echo(f"total_kdtree_bytes: {total_kdtree_bytes}")
