# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Configuration for VCAS and HCAS SafetyNet systems.

Ranges and strides are taken from the constants.jl files in
VerticalCAS/GenerateTable/mdp/ and HorizontalCAS/GenerateTable/mdp/.
"""

import math
from typing import Any

# VCAS configuration based on VerticalCAS/GenerateTable/mdp/constants.jl
VCAS_CONFIG: dict[str, Any] = {
    "name": "vcas",
    "function": "Vertical Collision Avoidance System",
    "description": (
        "SafetyNet for vertical collision avoidance in aircraft. "
        "Provides 9 advisories: COC, DNC, DND, DES1500, CL1500, "
        "SDES1500, SCL1500, SDES2500, SCL2500."
    ),
    "num_subsystems": 9,  # One subsystem per advisory region (ra=1-9)
    "num_inputs": 4,  # h, h_dot_own, h_dot_int, tau (s_adv is not included as input)
    "num_outputs": 9,  # 9 possible advisories/actions
    "datatype": "float32",
    # Input ranges from constants.jl with exact LinRange definitions
    "inputs": [
        {
            "id": "h",
            "name": "Relative altitude",
            "description": "Relative altitude between ownship and intruder",
            "unit": "ft",
            # hs = vcat of multiple LinRanges
            "ranges": [
                {"minimum": -8000.0, "maximum": -4000.0, "stride": 1000.0},
                {"minimum": -3000.0, "maximum": -1250.0, "stride": 250.0},
                {"minimum": -1000.0, "maximum": -800.0, "stride": 100.0},
                {"minimum": -700.0, "maximum": -150.0, "stride": 50.0},
                {"minimum": -100.0, "maximum": 100.0, "stride": 25.0},
                {"minimum": 150.0, "maximum": 700.0, "stride": 50.0},
                {"minimum": 800.0, "maximum": 1000.0, "stride": 100.0},
                {"minimum": 1250.0, "maximum": 3000.0, "stride": 250.0},
                {"minimum": 4000.0, "maximum": 8000.0, "stride": 1000.0},
            ],
        },
        {
            "id": "h_dot_own",
            "name": "Ownship vertical rate",
            "description": "Ownship vertical velocity",
            "unit": "ft/s",
            # vels = vcat of multiple LinRanges (same as ownship)
            "ranges": [
                {"minimum": -100.0, "maximum": -60.0, "stride": 10.0},
                {"minimum": -50.0, "maximum": -35.0, "stride": 5.0},
                {"minimum": -30.0, "maximum": 30.0, "stride": 3.0},
                {"minimum": 35.0, "maximum": 50.0, "stride": 5.0},
                {"minimum": 60.0, "maximum": 100.0, "stride": 10.0},
            ],
        },
        {
            "id": "h_dot_int",
            "name": "Intruder vertical velocity",
            "description": "Vertical velocity of intruder",
            "unit": "ft/s",
            # vels = vcat of multiple LinRanges (same as vo)
            "ranges": [
                {"minimum": -100.0, "maximum": -60.0, "stride": 10.0},
                {"minimum": -50.0, "maximum": -35.0, "stride": 5.0},
                {"minimum": -30.0, "maximum": 30.0, "stride": 3.0},
                {"minimum": 35.0, "maximum": 50.0, "stride": 5.0},
                {"minimum": 60.0, "maximum": 100.0, "stride": 10.0},
            ],
        },
        {
            "id": "tau",
            "name": "Time to closest approach",
            "description": "Time until closest approach between aircraft",
            "unit": "s",
            # taus = 0:1:40 (41 points)
            "ranges": [{"minimum": 0.0, "maximum": 40.0, "stride": 1.0}],
        },
        {
            "id": "s_adv",
            "name": "Previous advisory",
            "description": "Previous advisory issued (0-8, matching advisory indices)",
            "unit": "index",
            # s_adv = 0:8 (9 possible values)
            "ranges": [{"minimum": 0.0, "maximum": 8.0, "stride": 1.0}],
        },
    ],
    # Advisory indices for s_adv
    "advisories": {
        "COC": 0,
        "DNC": 1,
        "DND": 2,
        "DES1500": 3,
        "CL1500": 4,
        "SDES1500": 5,
        "SCL1500": 6,
        "SDES2500": 7,
        "SCL2500": 8,
    },
}


# HCAS configuration based on
# HorizontalCAS/GenerateTable/mdp/constants.jl
HCAS_CONFIG: dict[str, Any] = {
    "name": "hcas",
    "function": "Horizontal Collision Avoidance System",
    "description": (
        "SafetyNet for horizontal collision avoidance in aircraft. "
        "Provides 5 advisories: COC, WL, WR, SL, SR."
    ),
    "num_subsystems": 40,  # 5 actions (pra) x 8 tau values
    "num_inputs": 7,  # rho, theta, psi, v_own, v_int, tau, s_adv
    "num_outputs": 5,  # 5 possible advisories/actions
    "datatype": "float32",
    # Input ranges from constants.jl with exact definitions
    "inputs": [
        {
            "id": "rho",
            "name": "Range",
            "description": "Range to intruder",
            "unit": "ft",
            # RANGES array from constants.jl - exactly 32 specific
            # values. The near-range zone is non-uniformly spaced;
            # 25000 and 510 are isolated points between segments of
            # different strides.
            "ranges": [
                {"minimum": 0.0, "maximum": 100.0, "stride": 25.0},  # 0,25,50,75,100
                {"minimum": 150.0, "maximum": 200.0, "stride": 50.0},  # 150,200
                {"minimum": 300.0, "maximum": 500.0, "stride": 100.0},  # 300,400,500
                {"minimum": 510.0, "maximum": 510.0, "stride": 1.0},  # 510
                {"minimum": 750.0, "maximum": 1000.0, "stride": 250.0},  # 750,1000
                {"minimum": 1500.0, "maximum": 2000.0, "stride": 500.0},  # 1500,2000
                {
                    "minimum": 3000.0,
                    "maximum": 5000.0,
                    "stride": 1000.0,
                },  # 3000,4000,5000
                {
                    "minimum": 7000.0,
                    "maximum": 13000.0,
                    "stride": 2000.0,
                },  # 7000,9000,11000,13000
                {
                    "minimum": 15000.0,
                    "maximum": 21000.0,
                    "stride": 2000.0,
                },  # 15000,17000,19000,21000
                {"minimum": 25000.0, "maximum": 25000.0, "stride": 1.0},  # 25000
                {
                    "minimum": 30000.0,
                    "maximum": 40000.0,
                    "stride": 5000.0,
                },  # 30000,35000,40000
                {
                    "minimum": 48000.0,
                    "maximum": 56000.0,
                    "stride": 8000.0,
                },  # 48000,56000
            ],
        },
        {
            "id": "theta",
            "name": "Bearing",
            "description": "Bearing angle",
            "unit": "rad",
            # 41 points from -pi to pi
            "ranges": [
                {
                    "minimum": -math.pi,
                    "maximum": math.pi,
                    "stride": math.pi / 20,
                }
            ],
        },
        {
            "id": "psi",
            "name": "Relative heading",
            "description": "Relative heading",
            "unit": "rad",
            # 41 points from -pi to pi
            "ranges": [
                {
                    "minimum": -math.pi,
                    "maximum": math.pi,
                    "stride": math.pi / 20,
                }
            ],
        },
        {
            "id": "v_own",
            "name": "Ownship speed",
            "description": "Ownship horizontal speed (fixed at 200 ft/s in training)",
            "unit": "ft/s",
            # Fixed value in training data due to oneSpeed=True
            "ranges": [{"minimum": 200.0, "maximum": 200.0, "stride": 1.0}],
        },
        {
            "id": "v_int",
            "name": "Intruder speed",
            "description": "Intruder horizontal speed (fixed at 200 ft/s in training)",
            "unit": "ft/s",
            # Fixed value in training data due to oneSpeed=True
            "ranges": [{"minimum": 200.0, "maximum": 200.0, "stride": 1.0}],
        },
        {
            "id": "tau",
            "name": "Time to closest approach",
            "description": "Time until closest approach",
            "unit": "s",
            # Save at specific tau values: 0, 5, 10, 15, 20, 30, 40, 60
            "ranges": [
                {"minimum": 0.0, "maximum": 0.0, "stride": 1.0},
                {"minimum": 5.0, "maximum": 5.0, "stride": 1.0},
                {"minimum": 10.0, "maximum": 10.0, "stride": 1.0},
                {"minimum": 15.0, "maximum": 15.0, "stride": 1.0},
                {"minimum": 20.0, "maximum": 20.0, "stride": 1.0},
                {"minimum": 30.0, "maximum": 30.0, "stride": 1.0},
                {"minimum": 40.0, "maximum": 40.0, "stride": 1.0},
                {"minimum": 60.0, "maximum": 60.0, "stride": 1.0},
            ],
        },
        {
            "id": "s_adv",
            "name": "Previous advisory",
            "description": "Previous advisory issued (0-4, matching advisory indices)",
            "unit": "index",
            # s_adv = 0:4 (5 possible values)
            "ranges": [{"minimum": 0.0, "maximum": 4.0, "stride": 1.0}],
        },
    ],
    # Advisory indices for s_adv
    "advisories": {
        "COC": 0,
        "WL": 1,
        "WR": 2,
        "SL": 3,
        "SR": 4,
    },
}
