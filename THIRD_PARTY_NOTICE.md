<!--
SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>

SPDX-License-Identifier: MIT
-->

# Third-Party Notices

This repository includes code derived from third-party sources, in addition to code authored by the German Aerospace Center (DLR).

## HorizontalCAS / VerticalCAS

The Julia MDP solvers and Python training data generation scripts under [`HorizontalCAS/`](HorizontalCAS/) and [`VerticalCAS/`](VerticalCAS/) originate from:

- <https://github.com/sisl/HorizontalCAS>: Copyright (c) 2019 Stanford Intelligent Systems Laboratory
- <https://github.com/sisl/VerticalCAS>: Copyright (c) 2018 kjulian3

Both are licensed under the MIT License, as reflected in the `SPDX-FileCopyrightText` header of each individual file.
The canonical MIT license text is included at [`LICENSES/MIT.txt`](LICENSES/MIT.txt), per the [REUSE Specification 3.3](https://reuse.software/spec-3.3/).

These files are **vendored copies, not verbatim upstream snapshots**, which is why they are checked in rather than referenced as git submodules.
They have been modified: the sources were reduced to the parts this project builds on (the `PolicyViz` notebooks and the upstream Keras/TensorFlow training scripts for VerticalCAS were dropped, network training being handled by `castrainer`), REUSE-compliant SPDX headers were added, and the Python scripts were modernized (`pathlib`, formatting).
The upstream copyright and license of each file are unchanged.
