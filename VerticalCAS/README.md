<!--
SPDX-FileCopyrightText: 2018 kjulian3

SPDX-License-Identifier: MIT
-->
# VerticalCAS

Derived from [sisl/VerticalCAS](https://github.com/sisl/VerticalCAS).
This copy contains the two components needed by `castrainer`:

* Generate MDP table policy
* Format the table policy into neural network training data

> [!NOTE]
> The upstream Keras training script (`trainVertCAS.py`) and the `PolicyViz` visualization notebooks are **not** part of this repository.
> Network training is done by `castrainer` instead; see the [top-level README](../README.md).

## Generate MDP Policy

Required Julia Packages: Printf, POMDPs, POMDPModelTools, LocalFunctionApproximation, GridInterpolations, Distributed, SharedArrays, StaticArrays, HDF5

Tested with Julia v1.1.1 (pinned; see the [top-level README](../README.md) for why)

The policy is generated in parallel via Julia by running `julia -p NUM_PROCS SolveVCASMDP.jl` **inside** the `GenerateTable` folder, where NUM_PROCS is the number of processors you want to use.
The script resolves both its module path (`push!(LOAD_PATH, "mdp")`) and its output file relative to the current working directory, so it has to be started from that folder.
The top of `SolveVCASMDP.jl` specifies where the table should be written to as an HDF5 file (`VCAS.h5`).

## Generate Training Data

Required Python Packages: numpy, h5py

After generating the table, the table needs to be formatted into training data for the neural network.
To do this, run `python genTrainingData.py` in the GenerateNetworks folder.
The top of the file specifies the MDP table policy folder and the format for the resulting training data files.
Note that the table is split into separate files, one for each previous advisory, which allows separate networks to be trained for each previous advisory.

## Train Neural Network

Training is handled by `castrainer`, which replaces the upstream `trainVertCAS.py`:

```shell
# One network per previous advisory, plus LUTs and manifest
castrainer safetynet vcas --output-dir safetynet

# Or a single network from one training data file
castrainer train VerticalCAS/GenerateNetworks/VCAS_TrainingData_v5_01.h5
```

See the [top-level README](../README.md) for all options.
