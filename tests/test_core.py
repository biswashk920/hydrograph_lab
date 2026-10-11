import numpy as np
import pytest

from hydrograph_lab import (adjust_cn, compare, cumulative_runoff_mm,
                            kirpich_tc, retention_mm, run, storm,
                            unit_hydrograph)


def test_retention():
    assert retention_mm(75) == pytest.approx(25400 / 75 - 254)
    assert retention_mm(100) == pytest.approx(0)
    assert retention_mm(50) == pytest.approx(254)


def test_runoff_equation_textbook_form():
    # Q = (P - 0.2S)^2 / (P + 0.8S); P = 100 mm, CN = 75
    s = retention_mm(75)
    expected = (100 - 0.2 * s) ** 2 / (100 + 0.8 * s)
    assert cumulative_runoff_mm([100.0], 75)[0] == pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx(41.14, rel=1e-3)
    assert cumulative_runoff_mm([10.0], 75)[0] == 0          # below Ia = 0.2 S


def test_cn_100_runs_off_everything():
    assert cumulative_runoff_mm([37.0], 100)[0] == pytest.approx(37.0)


def test_soil_condition_adjustment():
    assert adjust_cn(75, "dry") == pytest.approx(55.75, rel=1e-3)
    assert adjust_cn(75, "wet") == pytest.approx(87.34, rel=1e-3)
    assert adjust_cn(75, "normal") == 75
    assert adjust_cn(100, "dry") == pytest.approx(100)
    assert adjust_cn(100, "wet") == pytest.approx(100)


def test_storm_shapes_conserve_depth():
    for shape in ("uniform", "front", "middle", "back", "scs24"):
        assert storm(shape, 100, 6, 0.1).sum() == pytest.approx(100, rel=1e-12)


def test_storm_shapes_look_right():
    front, back = storm("front", 100, 6, 0.1), storm("back", 100, 6, 0.1)
    mid = storm("middle", 100, 6, 0.1)
    n = front.size
    assert front[: n // 2].sum() > front[n // 2:].sum()
    assert back[: n // 2].sum() < back[n // 2:].sum()
    assert abs(int(mid.argmax()) - n // 2) <= 1
    assert storm("scs24", 100, 6, 0.1).size == 240            # forced to 24 h


def test_unit_hydrograph_peak_and_volume():
    u, tp, tb = unit_hydrograph(25, 2, 0.1)
    assert tp == pytest.approx(0.05 + 1.2)
    assert tb == pytest.approx(2.67 * tp)
    fine, _, _ = unit_hydrograph(25, 2, 0.01)                  # fine steps hit the peak
    assert fine.max() == pytest.approx(0.208 * 25 / (0.005 + 1.2), rel=0.01)
    assert u.sum() * 0.1 * 3600 == pytest.approx(25 * 1000, rel=1e-12)  # m3 per mm


def test_hand_calculated_peak():
    # One very short burst of 100 mm: peak = runoff depth x unit-hydrograph peak.
    r = run(25, 75, 2.0, [100.0], dt_hours=0.01)
    q_mm = 41.14
    qp_unit = 0.208 * 25 / (0.005 + 1.2)                       # 4.315 m3/s per mm
    assert r.peak_m3s == pytest.approx(q_mm * qp_unit, rel=0.01)   # about 177 m3/s
    assert r.time_to_peak_h == pytest.approx(1.205, abs=0.02)


def test_mass_balance():
    rain = storm("middle", 100, 6, 0.1)
    r = run(25, 75, 2.0, rain, 0.1, baseflow=3.0)
    assert r.runoff_volume_m3 == pytest.approx(r.runoff_depth_mm * 25 * 1000)
    direct = r.discharge - 3.0
    assert direct.sum() * 0.1 * 3600 == pytest.approx(r.runoff_volume_m3, rel=1e-9)
    assert r.runoff_coefficient == pytest.approx(r.runoff_depth_mm / 100)


def test_urbanization_raises_and_hastens_the_peak():
    rain = storm("middle", 100, 6, 0.1)
    before = run(25, 65, 2.5, rain, 0.1)
    after = run(25, 90, 1.2, rain, 0.1)
    c = compare(before, after)
    assert c["peak_change_pct"] > 0 and c["volume_change_pct"] > 0
    assert c["time_to_peak_shift_h"] < 0


def test_no_runoff_gives_baseflow_only():
    r = run(25, 75, 2.0, storm("uniform", 10, 6, 0.1), 0.1, baseflow=2.0)
    assert r.runoff_depth_mm == 0 and r.time_to_peak_h is None
    assert r.peak_m3s == pytest.approx(2.0)
    assert any("No runoff" in w for w in r.warnings)


def test_coarse_time_step_warning():
    r = run(25, 75, 0.5, [20.0] * 6, dt_hours=1.0)
    assert any("coarse" in w for w in r.warnings)


def test_kirpich():
    assert kirpich_tc(1000, 0.01) == pytest.approx(0.3907, rel=2e-3)


def test_bad_inputs_rejected():
    with pytest.raises(ValueError):
        run(25, 120, 2.0, [10.0])
    with pytest.raises(ValueError):
        run(25, 75, 2.0, [-1.0])
    with pytest.raises(ValueError):
        storm("hail", 100, 6, 0.1)
