# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT

FROM julia:1.1.1 AS base
WORKDIR /code
CMD ["/bin/sh"]
ENV JULIA_DEPOT_PATH=/juliadepot
RUN mkdir -p ${JULIA_DEPOT_PATH}
# Julia 1.1.1 is based on Debian 10 (Buster), which is EOL and has no more updates
# RUN apt-get update \
#     && apt-get -y upgrade \
#     && rm -rf /var/lib/apt/lists/*
RUN julia -e 'using Pkg; Pkg.add.([ \
    Pkg.PackageSpec(;name="ColorBrewer", version="v0.4"), \
    Pkg.PackageSpec(;name="Colors", version="v0.12.6"), \
    Pkg.PackageSpec(;name="Distributed"), \
    Pkg.PackageSpec(;name="GridInterpolations", version="v1.1.1"), \
    Pkg.PackageSpec(;name="HDF5", version="v0.12.5"), \
    Pkg.PackageSpec(;name="IJulia"), \
    Pkg.PackageSpec(;name="Interact", version="v0.10.3"), \
    Pkg.PackageSpec(;name="LocalFunctionApproximation", version="v1.1.0"), \
    Pkg.PackageSpec(;name="PGFPlots", version="v3.2.1"), \
    Pkg.PackageSpec(;name="POMDPModelTools", version="v0.1.2"), \
    Pkg.PackageSpec(;name="POMDPs", version="v0.7.0"), \
    Pkg.PackageSpec(;name="Printf"), \
    Pkg.PackageSpec(;name="Revise"), \
    Pkg.PackageSpec(;name="SharedArrays"), \
    Pkg.PackageSpec(;name="StaticArrays", version="v0.12.5"), \
    Pkg.PackageSpec(;name="WebIO", version="v0.8.15") \
    ])' \
    && rm -rf ${JULIA_DEPOT_PATH}/compiled

FROM base AS full
COPY README.md LICENSES/ CODE_OF_CONDUCT.md /code/
COPY HorizontalCAS /code/HorizontalCAS
COPY VerticalCAS /code/VerticalCAS
