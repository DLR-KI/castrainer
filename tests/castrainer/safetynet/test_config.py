# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for SafetyNet configuration."""

import math

from castrainer.safetynet.config import HCAS_CONFIG, VCAS_CONFIG


def test_vcas_name() -> None:
    """Test VCAS config name."""
    assert VCAS_CONFIG["name"] == "vcas"


def test_vcas_function() -> None:
    """Test VCAS config function name."""
    assert VCAS_CONFIG["function"] == "Vertical Collision Avoidance System"


def test_vcas_num_subsystems() -> None:
    """Test VCAS number of subsystems."""
    assert VCAS_CONFIG["num_subsystems"] == 9


def test_vcas_num_inputs() -> None:
    """Test VCAS number of network inputs (s_adv selects the subsystem, not a network input)."""
    assert VCAS_CONFIG["num_inputs"] == 4


def test_vcas_num_outputs() -> None:
    """Test VCAS number of outputs."""
    assert VCAS_CONFIG["num_outputs"] == 9


def test_vcas_datatype() -> None:
    """Test VCAS datatype."""
    assert VCAS_CONFIG["datatype"] == "float32"


def test_vcas_inputs_count() -> None:
    """Test VCAS has 5 inputs."""
    assert len(VCAS_CONFIG["inputs"]) == 5


def test_vcas_input_names() -> None:
    """Test VCAS input names."""
    expected_names = ["h", "h_dot_own", "h_dot_int", "tau", "s_adv"]
    actual_names = [inp["id"] for inp in VCAS_CONFIG["inputs"]]
    assert actual_names == expected_names


def test_vcas_h_range() -> None:
    """Test VCAS altitude range matches constants.jl."""
    h_input = VCAS_CONFIG["inputs"][0]
    assert math.isclose(min(r["minimum"] for r in h_input["ranges"]), -8000.0)
    assert math.isclose(max(r["maximum"] for r in h_input["ranges"]), 8000.0)


def test_vcas_velocity_range() -> None:
    """Test VCAS ownship velocity range matches constants.jl."""
    vo_input = VCAS_CONFIG["inputs"][1]
    assert math.isclose(min(r["minimum"] for r in vo_input["ranges"]), -100.0)
    assert math.isclose(max(r["maximum"] for r in vo_input["ranges"]), 100.0)


def test_vcas_intruder_velocity_range() -> None:
    """Test VCAS intruder velocity range matches constants.jl."""
    vi_input = VCAS_CONFIG["inputs"][2]
    assert math.isclose(min(r["minimum"] for r in vi_input["ranges"]), -100.0)
    assert math.isclose(max(r["maximum"] for r in vi_input["ranges"]), 100.0)


def test_vcas_tau_range() -> None:
    """Test VCAS tau range matches genTrainingData.py."""
    tau_input = VCAS_CONFIG["inputs"][3]
    ranges = tau_input["ranges"]
    assert math.isclose(min(r["minimum"] for r in ranges), 0.0)
    assert math.isclose(max(r["maximum"] for r in ranges), 40.0)


def test_hcas_name() -> None:
    """Test HCAS config name."""
    assert HCAS_CONFIG["name"] == "hcas"


def test_hcas_function() -> None:
    """Test HCAS config function name."""
    assert HCAS_CONFIG["function"] == "Horizontal Collision Avoidance System"


def test_hcas_num_subsystems() -> None:
    """Test HCAS number of subsystems (5 pra x 8 tau)."""
    assert HCAS_CONFIG["num_subsystems"] == 40


def test_hcas_num_inputs() -> None:
    """Test HCAS number of inputs."""
    assert HCAS_CONFIG["num_inputs"] == 7


def test_hcas_num_outputs() -> None:
    """Test HCAS number of outputs."""
    assert HCAS_CONFIG["num_outputs"] == 5


def test_hcas_datatype() -> None:
    """Test HCAS datatype."""
    assert HCAS_CONFIG["datatype"] == "float32"


def test_hcas_inputs_count() -> None:
    """Test HCAS has 7 inputs."""
    assert len(HCAS_CONFIG["inputs"]) == 7


def test_hcas_input_names() -> None:
    """Test HCAS input names."""
    expected_names = ["rho", "theta", "psi", "v_own", "v_int", "tau", "s_adv"]
    actual_names = [inp["id"] for inp in HCAS_CONFIG["inputs"]]
    assert actual_names == expected_names


def test_hcas_range_range() -> None:
    """Test HCAS range range matches constants.jl."""
    rho_input = HCAS_CONFIG["inputs"][0]
    assert math.isclose(min(r["minimum"] for r in rho_input["ranges"]), 0.0)
    assert math.isclose(max(r["maximum"] for r in rho_input["ranges"]), 56000.0)


def test_hcas_bearing_range() -> None:
    """Test HCAS bearing range matches constants.jl."""
    theta_input = HCAS_CONFIG["inputs"][1]
    import math

    assert abs(theta_input["ranges"][0]["minimum"] + math.pi) < 1e-10
    assert abs(theta_input["ranges"][0]["maximum"] - math.pi) < 1e-10


def test_hcas_heading_range() -> None:
    """Test HCAS relative heading range matches constants.jl."""
    psi_input = HCAS_CONFIG["inputs"][2]
    import math

    assert abs(psi_input["ranges"][0]["minimum"] + math.pi) < 1e-10
    assert abs(psi_input["ranges"][0]["maximum"] - math.pi) < 1e-10


def test_hcas_ownship_speed_range() -> None:
    """Test HCAS ownship speed matches constants.jl."""
    v_own_input = HCAS_CONFIG["inputs"][3]
    assert math.isclose(v_own_input["ranges"][0]["minimum"], 200.0)
    assert math.isclose(v_own_input["ranges"][0]["maximum"], 200.0)


def test_hcas_intruder_speed_range() -> None:
    """Test HCAS intruder speed matches constants.jl."""
    v_int_input = HCAS_CONFIG["inputs"][4]
    assert math.isclose(v_int_input["ranges"][0]["minimum"], 200.0)
    assert math.isclose(v_int_input["ranges"][0]["maximum"], 200.0)


def test_hcas_tau_range() -> None:
    """Test HCAS tau range matches constants.jl."""
    tau_input = HCAS_CONFIG["inputs"][5]
    ranges = tau_input["ranges"]
    assert math.isclose(min(r["minimum"] for r in ranges), 0.0)
    assert math.isclose(max(r["maximum"] for r in ranges), 60.0)


def test_vcas_input_structure() -> None:
    """Test VCAS input has required fields."""
    for inp in VCAS_CONFIG["inputs"]:
        assert "id" in inp
        assert "name" in inp
        assert "description" in inp
        assert "unit" in inp
        assert "ranges" in inp
        for r in inp["ranges"]:
            assert "minimum" in r
            assert "maximum" in r
            assert "stride" in r


def test_hcas_input_structure() -> None:
    """Test HCAS input has required fields."""
    for inp in HCAS_CONFIG["inputs"]:
        assert "id" in inp
        assert "name" in inp
        assert "description" in inp
        assert "unit" in inp
        assert "ranges" in inp
        for r in inp["ranges"]:
            assert "minimum" in r
            assert "maximum" in r
            assert "stride" in r


def test_vcas_range_stride_structure() -> None:
    """Test VCAS range stride structure."""
    for inp in VCAS_CONFIG["inputs"]:
        assert len(inp["ranges"]) > 0
        for r in inp["ranges"]:
            assert "minimum" in r
            assert "maximum" in r
            assert "stride" in r


def test_hcas_range_stride_structure() -> None:
    """Test HCAS range stride structure."""
    for inp in HCAS_CONFIG["inputs"]:
        assert len(inp["ranges"]) > 0
        for r in inp["ranges"]:
            assert "minimum" in r
            assert "maximum" in r
            assert "stride" in r
