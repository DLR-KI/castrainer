<!--
SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>

SPDX-License-Identifier: MIT
-->
# ACAS XA/XU Networks and the castrainer

[![The latest version of castrainer can be found on PyPI.](https://img.shields.io/pypi/v/castrainer.svg)](https://pypi.python.org/pypi/castrainer)
[![Information on what versions of Python castrainer supports can be found on PyPI.](https://img.shields.io/pypi/pyversions/castrainer.svg)](https://pypi.python.org/pypi/castrainer)
[![Python tests (pytest)](https://github.com/DLR-KI/castrainer/actions/workflows/pytest.yaml/badge.svg)](https://github.com/DLR-KI/castrainer/actions/workflows/pytest.yaml)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![prek](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json)](https://github.com/DLR-KI/castrainer)
[![REUSE status](https://api.reuse.software/badge/github.com/DLR-KI/castrainer)](https://api.reuse.software/info/github.com/DLR-KI/castrainer)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://conventionalcommits.org)

Train neural networks for HCAS and VCAS.

The repository contains the code to generate the LUTs for HCAS and VCAS and train the corresponding neural networks and Safety Nets with great flexibility.

[TOC]

## Install Julia

### `juliaup`

To install the correct version of Julia, use the [`juliaup`](https://github.com/JuliaLang/juliaup) script:

> The following two subsections regarding the installation of `juliaup` are copied from the `juliaup` README.

#### Windows

On Windows Julia and Juliaup can be installed directly from the [Windows store](https://www.microsoft.com/store/apps/9NJNWW8PVKMN).
One can also install exactly the same version by executing

```shell
winget install julia -s msstore
```

on a command line.

If the Windows Store is blocked on a system, we have an alternative [MSIX App Installer](https://learn.microsoft.com/en-us/windows/msix/app-installer/app-installer-file-overview) based setup.
Note that this is currently experimental, please report back successes and failures [on GitHub](https://github.com/JuliaLang/juliaup/issues/343).
To use the App Installer version, download [this](https://install.julialang.org/Julia.appinstaller) file and open it by double clicking on it.

#### Mac and Linux

Juliaup can be installed on Linux or Mac by executing

```shell
curl -fsSL https://install.julialang.org | sh
```

in a shell.

### Julia

In a shell run

```shell
juliaup add 1.1.1 && juliaup default 1.1.1
```

> [!NOTE]
> Julia 1.1.1 reached end of life long ago, and so has the Debian 10 (Buster) base of the [`Dockerfile`](Dockerfile) that provides it.
> It is pinned deliberately: the MDP solvers under [`HorizontalCAS/`](HorizontalCAS/) and [`VerticalCAS/`](VerticalCAS/) are built on the POMDPs.jl v0.7 / POMDPModelTools.jl v0.1 API, which was removed in later releases, so the solver code does not run on a current Julia.
> The pin therefore only affects the *table generation* step, which is run once, offline, on trusted inputs.
> Everything downstream of it---`castrainer` and the whole Python toolchain---is on current, supported versions and never needs Julia.

### Required packages

This project requires the following Julia packages (the list matches the [`Dockerfile`](Dockerfile); `Printf` through `HDF5` are what the MDP solvers themselves need, the remainder are carried over from the upstream plotting tooling):

- Printf
- POMDPs@v0.7.0
- POMDPModelTools@v0.1.2
- LocalFunctionApproximation@v1.1.0
- GridInterpolations@v1.1.1
- Distributed
- SharedArrays
- StaticArrays@v0.12.5
- HDF5@v0.12.5
- Revise
- Interact@v0.10.3
- PGFPlots@v3.2.1
- Colors@v0.12.6
- ColorBrewer@v0.4
- IJulia
- WebIO@v0.8.15

To install these packages, run the following command in a shell:

```shell
julia -e 'using Pkg; Pkg.add.([Pkg.PackageSpec(;name="Printf"), Pkg.PackageSpec(;name="POMDPs", version="v0.7.0"), Pkg.PackageSpec(;name="POMDPModelTools", version="v0.1.2"), Pkg.PackageSpec(;name="LocalFunctionApproximation", version="v1.1.0"), Pkg.PackageSpec(;name="GridInterpolations", version="v1.1.1"), Pkg.PackageSpec(;name="Distributed"), Pkg.PackageSpec(;name="SharedArrays"), Pkg.PackageSpec(;name="StaticArrays", version="v0.12.5"), Pkg.PackageSpec(;name="HDF5", version="v0.12.5"), Pkg.PackageSpec(;name="Revise"), Pkg.PackageSpec(;name="Interact", version="v0.10.3"), Pkg.PackageSpec(;name="PGFPlots", version="v3.2.1"), Pkg.PackageSpec(;name="Colors", version="v0.12.6"), Pkg.PackageSpec(;name="ColorBrewer", version="v0.4"), Pkg.PackageSpec(;name="IJulia"), Pkg.PackageSpec(;name="WebIO", version="v0.8.15")])'
```

## Generating the LUTs

Both solvers resolve their module path (`push!(LOAD_PATH, "mdp")`) and their output file relative to the *current working directory*, so they have to be started from inside their own `GenerateTable` folder:

```shell
# Horizontal CAS -> writes HorizontalCAS/GenerateTable/HCAS.h5
cd HorizontalCAS/GenerateTable && julia -p NPROC SolveHCASMDP.jl

# Vertical CAS -> writes VerticalCAS/GenerateTable/VCAS.h5
cd VerticalCAS/GenerateTable && julia -p NPROC SolveVCASMDP.jl
```

`NPROC` is the number of worker processes to use.

## Generating the Training Data

The `.h5` files the `castrainer` examples below refer to are **not** part of the repository or of the installed package: together they are roughly 3.7 GB, and they are derived artefacts of the LUTs generated above.
Create them with the two generator scripts, which read the `HCAS.h5`/`VCAS.h5` tables from `../GenerateTable/` and write one training file per subsystem next to themselves:

```shell
python HorizontalCAS/GenerateNetworks/genTrainingData.py
python VerticalCAS/GenerateNetworks/genTrainingData.py
```

This produces:

```text
HorizontalCAS/GenerateNetworks/HCAS_rect_TrainingData_v6_pra{0..4}_tau{00,05,10,15,20,30,40,60}.h5   #  40 files, ~132 MB
VerticalCAS/GenerateNetworks/VCAS_TrainingData_v5_{01..09}.h5                                        #   9 files, ~3.6 GB
```

`castrainer safetynet` looks for exactly this layout.
By default it resolves it against the current working directory (i.e. it just works from a repository checkout); use `--data-dir` or the `CASTRAINER_DATA_DIR` environment variable to point it somewhere else, e.g. at a scratch filesystem:

```shell
castrainer safetynet all --data-dir /scratch/acasxu
export CASTRAINER_DATA_DIR=/scratch/acasxu   # equivalent, useful in job scripts
```

The `train` and `study` commands take the `.h5` files as plain arguments instead, so they accept any path.

> [!TIP]
> Pre-generated HorizontalCAS training data and trained networks are also published by the upstream project; see the links in [`HorizontalCAS/README.md`](HorizontalCAS/README.md).

## Installing castrainer

Install the package using uv:

```shell
uv sync
```

## CLI Commands

The `castrainer` CLI provides the following commands:

| Command     | Description                                                    |
| ----------- | -------------------------------------------------------------- |
| `study`     | Train models with hyperparameter study (multiple combinations) |
| `train`     | Train a single neural network and corresponding LUT            |
| `safetynet` | Create a full safety net including all JSON files              |
| `evaluate`  | Convert JSON result files to CSV                               |
| `validate`  | Validate SafetyNet directory structure and files               |
| `info`      | Show information about available systems and configurations    |
| `load`      | Load SafetyNet from JSON files into memory                     |
| `infer`     | Benchmark SafetyNet inference time                             |

Every option below is listed as `castrainer <command> --help` reports it; boolean options are shown with both of their flag forms.

### Study Command

Train models with multiple hyperparameter combinations:

```bash
# Basic study with default parameters
castrainer study VerticalCAS/GenerateNetworks/VCAS_TrainingData_v5_01.h5

# With custom hyperparameters
castrainer study VerticalCAS/GenerateNetworks/*.h5 \
    --activations relu,gelu \
    --hidden-nodes 25,50 \
    --hidden-layers 2,3

# Parallel training with multiple GPUs
castrainer study VerticalCAS/GenerateNetworks/*.h5 \
    --strategy parallel \
    --min-free-gpu-memory 3.0 \
    --devices 0,1

# Resume training (skip completed combinations)
castrainer study VerticalCAS/GenerateNetworks/*.h5 --resume

# With experiment tracking
castrainer study VerticalCAS/GenerateNetworks/*.h5 --experiment-name my_experiment

# Real-world example: VCAS hyperparameter search across 16 parallel GPU jobs
castrainer study VerticalCAS/GenerateNetworks/VCAS_TrainingData_v5_01.h5 \
    --strategy parallel \
    --nproc 16 \
    --min-free-gpu-memory 8.0 \
    --activations relu,gelu \
    --hidden-nodes 10,15,20 \
    --hidden-layers 2,3 \
    --batch-size 128 \
    --max-epochs 5000 \
    --patience 200 \
    --enable-checkpointing \
    --enable-progress-bar \
    --experiment-name vcas_optimized
```

#### Study Options

| Option                                          | Description                                                              | Default                 |
| ----------------------------------------------- | ------------------------------------------------------------------------ | ----------------------- |
| `FILE...`                                       | HDF5 file(s) or directory path(s); supports glob patterns like `*.h5`    | - (required)            |
| `--nproc`                                       | Number of processes for data loading                                     | 8                       |
| `--one-hot` / `--no-one-hot`                    | Use one-hot encoding for targets                                         | `--one-hot`             |
| `--strategy`                                    | Training strategy (`sequential` or `parallel`)                           | sequential              |
| `--min-free-gpu-memory`                         | Minimum free GPU memory in GB required to start a job                    | 3.0                     |
| `--devices`                                     | GPU devices to monitor (comma-separated)                                 | 0                       |
| `--experiment-name`                             | Experiment name for result organization                                  | default                 |
| `--resume` / `--no-resume`                      | Skip already completed hyperparameter combinations                       | `--no-resume`           |
| `--activations`                                 | Comma-separated activation functions                                     | relu,leakyrelu,gelu     |
| `--hidden-nodes`                                | Comma-separated hidden node counts                                       | 25,50,100,150,200       |
| `--hidden-layers`                               | Comma-separated hidden layer counts                                      | 2,3,5,7                 |
| `--batch-size`                                  | Batch size for training                                                  | 32                      |
| `--max-epochs`                                  | Maximum epochs                                                           | 10000                   |
| `--patience`                                    | Early stopping patience                                                  | 1000                    |
| `--enable-checkpointing` / `--no-checkpointing` | Enable model checkpointing                                               | `--no-checkpointing`    |
| `--enable-progress-bar` / `--no-progress-bar`   | Enable progress bar                                                      | `--enable-progress-bar` |
| `--config-file`                                 | YAML/JSON config file for hyperparameters (see [`examples/`](examples/)) | -                       |
| `--num-jobs`                                    | Number of job chunks to run                                              | -                       |
| `--job-id`                                      | Job chunk ID (0-based) for parallelization                               | -                       |
| `--seed`                                        | Random seed; reproducible only together with `--strategy sequential`     | -                       |

### Train Command

Train a single neural network configuration:

```bash
castrainer train VerticalCAS/GenerateNetworks/VCAS_TrainingData_v5_01.h5 \
    --activation relu \
    --hidden-nodes 100 \
    --hidden-layers 4 \
    --max-epochs 10000 \
    --output-dir trained_models
```

#### Train Options

| Option                                          | Description                                                | Default                 |
| ----------------------------------------------- | ---------------------------------------------------------- | ----------------------- |
| `FILE`                                          | HDF5 file for training                                     | - (required)            |
| `--activation`                                  | Activation function (relu, leakyrelu, gelu, tanh, sigmoid) | relu                    |
| `--hidden-nodes`                                | Number of hidden nodes per layer                           | 100                     |
| `--hidden-layers`                               | Number of hidden layers                                    | 4                       |
| `--batch-size`                                  | Batch size for training                                    | 32                      |
| `--max-epochs`                                  | Maximum epochs                                             | 10000                   |
| `--patience`                                    | Early stopping patience                                    | 1000                    |
| `--nproc`                                       | Number of processes for data loading                       | 8                       |
| `--one-hot` / `--no-one-hot`                    | Use one-hot encoding for targets                           | `--no-one-hot`          |
| `--enable-checkpointing` / `--no-checkpointing` | Enable model checkpointing                                 | `--no-checkpointing`    |
| `--enable-progress-bar` / `--no-progress-bar`   | Enable progress bar                                        | `--enable-progress-bar` |
| `--output-dir`                                  | Output directory for trained model                         | trained                 |
| `--seed`                                        | Random seed for reproducible training                      | -                       |

### SafetyNet Command

Create a complete safety net.
**VCAS and HCAS are independent systems**, each with its own manifest file.

Unlike `train` and `study`, this command does not take file arguments: it trains every subsystem of a system and locates the training data itself (see [Generating the Training Data](#generating-the-training-data)).

```bash
# Train both VCAS and HCAS (creates independent manifests)
castrainer safetynet all --output-dir safetynet

# Train only VCAS
castrainer safetynet vcas --output-dir safetynet

# Train only HCAS
castrainer safetynet hcas --output-dir safetynet

# Training data outside the repository
castrainer safetynet all --data-dir /scratch/acasxu

# Train with custom parameters
castrainer safetynet all \
    --output-dir safetynet \
    --max-epochs 10000 \
    --hidden-nodes 100 \
    --hidden-layers 4
```

#### SafetyNet Options

| Option                                          | Description                                                                | Default                                            |
| ----------------------------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------- |
| `SYSTEM...`                                     | Systems to train: `vcas`, `hcas`, or `all`                                 | - (required)                                       |
| `--output-dir`                                  | Output directory for trained models                                        | safetynet                                          |
| `--data-dir`                                    | Directory holding the generated `HorizontalCAS/` and `VerticalCAS/` data   | `$CASTRAINER_DATA_DIR`, else the current directory |
| `--activations`                                 | Activation functions to use (only the first is used; use `study` to sweep) | relu                                               |
| `--hidden-nodes`                                | Number of hidden nodes (only the first is used; use `study` to sweep)      | 100                                                |
| `--hidden-layers`                               | Number of hidden layers                                                    | 4                                                  |
| `--batch-size`                                  | Batch size for training                                                    | 32                                                 |
| `--max-epochs`                                  | Maximum number of epochs                                                   | 10000                                              |
| `--patience`                                    | Early stopping patience                                                    | 1000                                               |
| `--enable-progress-bar` / `--no-progress-bar`   | Enable progress bar                                                        | `--enable-progress-bar`                            |
| `--enable-checkpointing` / `--no-checkpointing` | Enable model checkpointing                                                 | `--no-checkpointing`                               |
| `--one-hot` / `--no-one-hot`                    | Use one-hot encoding for targets                                           | `--no-one-hot`                                     |
| `--nproc`                                       | Number of processes for data loading                                       | 8                                                  |
| `--seed`                                        | Random seed; reproducible only with the (default) sequential strategy      | -                                                  |

### Evaluate Command

Convert the JSON result files written by `study` into CSV files.
One CSV is written per case type, activation function and grouping (e.g. `data_VCAS_ReLU_layer_size.csv`), which is why the command takes an output *directory* rather than a single file:

```bash
castrainer evaluate results/my_experiment/20260820_120000 --output-dir data
```

#### Evaluate Options

| Option         | Description                            | Default      |
| -------------- | -------------------------------------- | ------------ |
| `INPUT_DIR`    | Directory containing JSON result files | - (required) |
| `--output-dir` | Output directory for CSV files         | data         |

### Validate Command

Validate a safety net directory.
The JSON schemas are shipped inside the package (see [`src/castrainer/schemas/`](src/castrainer/schemas/)), so no schema paths need to be passed; supply them only to validate against a modified or newer schema.
Each system is validated independently, so point the command at a system directory:

```bash
# Uses the bundled schemas
castrainer validate safetynet/vcas

# Override one or more schemas
castrainer validate safetynet/vcas \
    --manifest-schema src/castrainer/schemas/manifest.schema.json \
    --safetynet-schema src/castrainer/schemas/safetynet.schema.json \
    --nnet-schema src/castrainer/schemas/nnet.schema.json
```

Exits with code 0 on success and 1 on failure.

#### Validate Options

| Option               | Description                                                                                                            | Default        |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------- | -------------- |
| `SAFETYNET_DIR`      | Directory containing SafetyNet files                                                                                   | - (required)   |
| `--manifest-schema`  | Path to manifest schema file                                                                                           | bundled schema |
| `--safetynet-schema` | Path to SafetyNet (LUT) schema file                                                                                    | bundled schema |
| `--nnet-schema`      | Path to nnet schema file, used to validate JSON-format (`jnet`) networks; `.pt`/`.pth`/`.onnx` networks are unaffected | bundled schema |

### Info Command

Print the available systems and their configuration (number of subsystems, inputs, outputs and the advisories each system provides).
Takes no arguments or options:

```bash
castrainer info
```

### Load Command

Load a SafetyNet from its JSON files into memory and report what was loaded---useful as a quick sanity check that a generated safety net is complete and readable:

```bash
# Load every system found in the directory
castrainer load safetynet

# Load a single system
castrainer load safetynet --system vcas
```

#### Load Options

| Option          | Description                          | Default                |
| --------------- | ------------------------------------ | ---------------------- |
| `SAFETYNET_DIR` | Directory containing SafetyNet files | - (required)           |
| `--system`      | System to load (`vcas` or `hcas`)    | all discovered systems |

### Infer Command

Benchmark SafetyNet inference time (network and LUT lookups):

```bash
castrainer infer safetynet --system vcas --repetitions 100
```

#### Infer Options

| Option                         | Description                             | Default                |
| ------------------------------ | --------------------------------------- | ---------------------- |
| `SAFETYNET_DIR`                | Directory containing SafetyNet files    | - (required)           |
| `--system`                     | System to benchmark (`vcas` or `hcas`)  | all discovered systems |
| `--repetitions`                | Number of benchmark repetitions         | 100                    |
| `--warmup`                     | Number of warmup runs                   | 10                     |
| `--sample-size`                | Number of LUT hits to sample per bundle | 5000                   |
| `--cpu-only` / `--no-cpu-only` | Force CPU-only benchmarking             | `--no-cpu-only`        |

## Full Workflow: Create and Validate Safety Net

### Step 1: Generate Training Data

Solve the MDPs, then convert the resulting tables into HDF5 training data.
The Julia solvers must be started from inside their own `GenerateTable` folder (see [Generating the LUTs](#generating-the-luts)):

```bash
# Generate HCAS training data
(cd HorizontalCAS/GenerateTable && julia -p 5 SolveHCASMDP.jl)
python HorizontalCAS/GenerateNetworks/genTrainingData.py

# Generate VCAS training data
(cd VerticalCAS/GenerateTable && julia -p 5 SolveVCASMDP.jl)
python VerticalCAS/GenerateNetworks/genTrainingData.py
```

### Step 2: Train Safety Net

```bash
# Train complete safety net
castrainer safetynet all --output-dir safetynet

# Or train individual systems
castrainer safetynet vcas --output-dir safetynet/vcas
castrainer safetynet hcas --output-dir safetynet/hcas
```

### Step 3: Evaluate Results

```bash
# Convert results to CSV for analysis
castrainer evaluate results/default/<YYYYMMDD_HHMMSS> --output-dir data
```

### Step 4: Validate Safety Net

```bash
# Validate the generated safety net, one system at a time
castrainer validate safetynet/vcas
castrainer validate safetynet/hcas
```

## Output Structure

After training, files are organized as follows:

```text
safetynet/
    vcas/
        vcas_01.pt              # Trained network
        vcas_01_lut.json        # Lookup table
        vcas.json               # VCAS manifest (independent)
        vcas_01_training_info.json  # Training info with sizes, loss, duration
    hcas/
        hcas_pra0_tau00.pt      # Trained network
        hcas_pra0_tau00_lut.json # Lookup table
        hcas.json               # HCAS manifest (independent)
        hcas_pra0_tau00_training_info.json  # Training info

results/
    <experiment-name>/
        <YYYYMMDD_HHMMSS>/
            status.json           # Tracking file
            *.json                # Individual results

data/                             # Default --output-dir of `castrainer evaluate`
    data_VCAS_ReLU_layer_size.csv
    data_VCAS_ReLU_n_hidden_layers.csv
    ...
```

## Activation Functions

Supported activation functions:

- `relu` - ReLU (default)
- `leakyrelu` - Leaky ReLU
- `gelu` - GELU
- `tanh` - Tanh
- `sigmoid` - Sigmoid

## System Requirements

- Python 3.12 or higher
- CUDA-compatible GPU (for parallel training)
- NVIDIA Management Library (pynvml) for GPU memory monitoring

## Troubleshooting

### GPU Memory Issues

If training crashes due to insufficient GPU memory:

1. Use `--strategy sequential` instead of `parallel`
2. Increase `--min-free-gpu-memory` threshold
3. Reduce `--batch-size`
4. Train fewer combinations in parallel

### Resuming Training

Use `--resume` to skip already completed hyperparameter combinations:

```bash
castrainer study VerticalCAS/GenerateNetworks/*.h5 --resume
```

### CPU Thread Limits

Thread limits are set automatically via `--nproc`.
No manual environment variable setup needed.

## Citation

If you use this software, please cite the paper it accompanies:

> J. M. Christensen, T. Stefani, E. Hoemann, F. Köster, and S. Hallerbach, "On the Applicability of Safety Nets: A Safety-By-Design Solution for Certifying Neural Networks," in *35th Congress of the International Council of the Aeronautical Sciences (ICAS)*, Sydney, Australia, 2026, pp. 1--27.
> doi: [10.71945/icas2026_0210](https://doi.org/10.71945/icas2026_0210)

```bibtex
@inproceedings{Christensen2026,
  author = {Christensen, Johann Maximilian and Stefani, Thomas and Hoemann, Elena and K\"{o}ster, Frank and Hallerbach, Sven},
  publisher = {CEAS},
  address = {Sydney, Australia},
  booktitle = {35th Congress of the International Council of the Aeronautical Sciences ({ICAS})},
  doi = {10.71945/icas2026_0210},
  eventdate = {2026-09-13/2026-09-18},
  issn = {2958-4647},
  month = {09},
  pages = {1--27},
  title = {On the Applicability of Safety Nets: A Safety-By-Design Solution for Certifying Neural Networks},
  year = {2026},
}
```

To cite the software itself instead, see [`CITATION.cff`](CITATION.cff), which lists the paper above as its `preferred-citation`.

## License

Copyright and license information are provided in accordance with the [REUSE Specification 3.3](https://reuse.software/spec-3.3/).
Code authored by DLR is licensed under the MIT License (see [`LICENSE`](LICENSE)).
The `HorizontalCAS/` and `VerticalCAS/` directories additionally include third-party code; see [`THIRD_PARTY_NOTICE.md`](THIRD_PARTY_NOTICE.md) for details.
