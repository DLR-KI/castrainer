<!--
SPDX-FileCopyrightText: 2019 Stanford Intelligent Systems Laboratory

SPDX-License-Identifier: MIT
-->
# HorizontalCAS

This repository describes how to generate HorizontalCAS score tables and train a neural network representation.
HorizontalCAS is a simple, notional collision avoidance system (CAS) that gives horizontal turning advisories to an aircraft to avoid an intruder.
This system is inspired by early prototypes of ACAS Xu and neural networks trained to represent the score table, but HorizontalCAS is not related in any way to ACAS.
This simple system is intended to facilitate research towards safety-critical neural networks and their verification in the hopes that lessons learned can be applied to real systems.

This repository supports a paper presented at the Digital Avionics Systems Conference (DASC) in 2019, which can be found [on arXiv](https://arxiv.org/pdf/1912.07084.pdf).

> [!NOTE]
> This is a copy of [sisl/HorizontalCAS](https://github.com/sisl/HorizontalCAS), reduced to the parts `castrainer` builds on.
> The upstream `PolicyViz` visualization notebooks are **not** included.
> `trainHCAS.py` (TensorFlow) is kept for reference only---network training is done by `castrainer`; see the [top-level README](../README.md).

If HorizontalCAS is useful for your research, please cite

```bibtex
@inproceedings{julian2019guaranteeing,
    title={Guaranteeing safety for neural network-based aircraft collision avoidance systems},
    author={Julian, Kyle D and Kochenderfer, Mykel J},
    booktitle={Digital Avionics Systems Conference (DASC)},
    year={2019}
}
```

## Saved training data and neural networks

Although this repository contains the source code used to generate the advisory score table and train a neural network representation, saved copies of training data and trained neural networks can be found here:

* [**Training Data**](https://drive.google.com/drive/folders/14kcGM_G5sq72BpCfD4dimp27S7ael3by?usp=sharing)
* [**Trained Networks**](https://drive.google.com/drive/folders/1Sj2noNh65xbG6H1fO3DkS1GnevSYTa5b?usp=sharing)

The remainder of this README describes how the score table is generated in Julia and how it is turned into neural network training data.

## Generate MDP Policy

Required Julia Packages: Printf, POMDPs@v0.7.0, POMDPModelTools@v0.1.2, LocalFunctionApproximation, GridInterpolations, Distributed, SharedArrays, StaticArrays, HDF5

Tested with Julia v1.1.1 (pinned; see the [top-level README](../README.md) for why)
> Note: A Docker container with Julia v1.1.1 and the dependencies set up at the correct versions is available with the Dockerfile of this repository.
> Have a look at the end of this document on a quick guide on how to use it.

The policy is generated in parallel via Julia by running `julia -p NUM_PROCS SolveHCASMDP.jl` **inside** the GenerateTable folder, where NUM_PROCS is the number of processors you want to use.
The script resolves both its module path (`push!(LOAD_PATH, "mdp")`) and its output file relative to the current working directory, so it has to be started from that folder.
The top of `SolveHCASMDP.jl` specifies where the table should be written to as an HDF5 file (`HCAS.h5`).

## Generate Training Data

Required Python Packages: numpy, h5py

After generating the table, the table needs to be formatted into training data for the neural network.
To do this, run `python genTrainingData.py` in the GenerateNetworks folder.
The top of the file specifies the MDP table policy folder and the format for the resulting training data files.
Note that the table is split into separate files, one for each previous advisory and tau combinations, which allows separate networks to be trained for each combination.

## Train Neural Networks

Training is handled by `castrainer`:

```shell
# One network per (previous advisory, tau) pair, plus LUTs and manifest
castrainer safetynet hcas --output-dir safetynet

# Or a single network from one training data file
castrainer train HorizontalCAS/GenerateNetworks/HCAS_rect_TrainingData_v6_pra0_tau00.h5
```

See the [top-level README](../README.md) for all options.

The original TensorFlow script is still present as `trainHCAS.py` (`python trainHCAS.py PREV_ADV TAU <gpu_ind>`, `requirements.txt` pins its dependencies), but it is unmaintained here and expects the upstream `./TrainingData/` layout rather than the files `genTrainingData.py` writes next to itself.

## Julia Docker Container

The Dockerfile of this repository contains a docker container which contains Julia v1.1 and the packages that are required to run the Julia code in this repository.
To use the container, install docker and run the following commands from the root directory of this repository:

```shell
docker build . -t hcas
docker run -it --rm --mount src="$PWD",target=/code,type=bind hcas bash
```

This will start bash inside the container.
To leave the container type "exit".
To execute the code in this repository, just execute the necessary commands inside this bash.
The outputs will be available in your copy of the repository (facilitated via the --mount option to docker run).
