"""Rainfall-runoff hydrograph: SCS Curve Number losses + SCS triangular unit hydrograph.

Units everywhere: rainfall and runoff in mm, area in km2, time in hours,
discharge in m3/s.

Method
------
1. Losses (SCS Curve Number):  S = 25400/CN - 254 (mm),  Ia = ia_ratio * S,
   Q = (P - Ia)^2 / (P - Ia + S)  for P > Ia, else 0   (ia_ratio = 0.2 gives
   the textbook Q = (P - 0.2 S)^2 / (P + 0.8 S)).  Applied to the cumulative
   rainfall; the runoff of each time step is the difference between steps.
2. Catchment response (SCS triangular unit hydrograph):
   lag = 0.6 tc,  Tp = dt/2 + lag,  Qp = 0.208 A / Tp  (m3/s per mm of runoff),
   base time = 2.67 Tp.  Ordinates are scaled slightly so that the volume is
   exactly A * 1000 m3 per mm of runoff (mass balance).
3. Hydrograph: convolution of the runoff of each time step with the unit
   hydrograph, plus a constant baseflow.

The catchment is treated as one lump and the response as linear.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

STORM_SHAPES = ("uniform", "front", "middle", "back", "scs24")
CONDITIONS = ("dry", "normal", "wet")

# SCS Type II 24-hour storm: cumulative fraction of the 24-h depth at given hours
# (points of the NRCS table, interpolated linearly between them).
_TYPE_II = (
    (0, 0.0), (2, 0.022), (4, 0.048), (6, 0.080), (7, 0.098), (8, 0.120),
    (8.5, 0.133), (9, 0.147), (9.5, 0.163), (9.75, 0.172), (10, 0.181),
    (10.5, 0.204), (11, 0.235), (11.5, 0.283), (11.75, 0.357), (12, 0.663),
    (12.5, 0.735), (13, 0.772), (13.5, 0.799), (14, 0.820), (16, 0.880),
    (20, 0.952), (24, 1.0),
)


# ---------------------------------------------------------------- losses
def retention_mm(cn: float) -> float:
    """Potential maximum retention S (mm) for a curve number."""
    if not 1 <= cn <= 100:
        raise ValueError("curve number must be between 1 and 100")
    return 25400.0 / cn - 254.0


def adjust_cn(cn: float, condition: str = "normal") -> float:
    """Curve number for dry (AMC I), normal (AMC II) or wet (AMC III) soil."""
    if condition == "normal":
        return cn
    if condition == "dry":
        return 4.2 * cn / (10 - 0.058 * cn)
    if condition == "wet":
        return 23 * cn / (10 + 0.13 * cn)
    raise ValueError(f"condition must be one of {CONDITIONS}")


def cumulative_runoff_mm(p_cum, cn: float, ia_ratio: float = 0.2):
    """Cumulative runoff (mm) for cumulative rainfall (mm)."""
    p = np.asarray(p_cum, dtype=float)
    s = retention_mm(cn)
    excess = np.maximum(p - ia_ratio * s, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(excess > 0, excess**2 / (excess + s), 0.0)
    return q


# ---------------------------------------------------------------- storms
def storm(shape: str, depth_mm: float, duration_h: float, dt_h: float):
    """Rainfall depth (mm) in each time step; the steps add up to depth_mm.

    uniform, front (heavy at the start), middle (peak at the centre) and back
    (heavy at the end) are smooth shapes; scs24 is the SCS Type II 24-hour
    design storm (duration is fixed at 24 h).
    """
    if depth_mm < 0 or dt_h <= 0:
        raise ValueError("depth must be >= 0 and dt must be > 0")
    if shape == "scs24":
        duration_h = 24.0
    n = max(1, int(round(duration_h / dt_h)))
    x = np.linspace(0.0, 1.0, n + 1)
    if shape == "uniform":
        f = x
    elif shape == "front":
        f = 1 - (1 - x) ** 2
    elif shape == "back":
        f = x**2
    elif shape == "middle":
        f = np.where(x <= 0.5, 2 * x**2, 1 - 2 * (1 - x) ** 2)
    elif shape == "scs24":
        hours, frac = zip(*_TYPE_II)
        f = np.interp(x * 24.0, hours, frac)
    else:
        raise ValueError(f"shape must be one of {STORM_SHAPES}")
    return depth_mm * np.diff(f)


# ------------------------------------------------------- unit hydrograph
def kirpich_tc(length_m: float, slope: float) -> float:
    """Time of concentration (hours) from channel length (m) and slope (m/m)."""
    return 0.0195 * length_m**0.77 * slope**-0.385 / 60.0


def unit_hydrograph(area_km2: float, tc_h: float, dt_h: float):
    """SCS triangular unit hydrograph for runoff of duration dt.

    Returns (u, tp, tb): u[k] is the discharge (m3/s per mm of runoff) at time
    k*dt (u[0] = 0), tp the time to peak and tb the base time, both in hours.
    """
    if area_km2 <= 0 or tc_h <= 0 or dt_h <= 0:
        raise ValueError("area, tc and dt must be positive")
    tp = dt_h / 2 + 0.6 * tc_h
    tb = 2.67 * tp
    qp = 0.208 * area_km2 / tp
    k = int(math.ceil(tb / dt_h))
    t = np.arange(k + 1) * dt_h
    u = np.where(t <= tp, qp * t / tp,
                 np.where(t < tb, qp * (tb - t) / (tb - tp), 0.0))
    u *= area_km2 * 1000.0 / (u.sum() * dt_h * 3600.0)   # exact mass balance
    return u, tp, tb


# ----------------------------------------------------------- hydrograph
@dataclass(frozen=True)
class HydrographResult:
    dt_h: float
    time_h: np.ndarray          # 0, dt, 2dt, ...
    discharge: np.ndarray       # m3/s, baseflow included
    rain_mm: np.ndarray         # rainfall in each storm step
    runoff_mm: np.ndarray       # runoff in each storm step
    cn_used: float
    retention_mm: float
    initial_abstraction_mm: float
    rain_total_mm: float
    runoff_depth_mm: float
    runoff_volume_m3: float
    runoff_coefficient: float
    peak_m3s: float
    time_to_peak_h: Optional[float]   # None when there is no runoff
    lag_h: float
    tp_h: float
    base_time_h: float
    unit_peak: float            # m3/s per mm
    warnings: tuple


def run(area_km2, cn, tc_hours, rain_mm, dt_hours=0.1, condition="normal",
        ia_ratio=0.2, baseflow=0.0) -> HydrographResult:
    """Flood hydrograph for a storm (rainfall in mm per time step)."""
    rain = np.asarray(rain_mm, dtype=float)
    if rain.size == 0 or np.any(rain < 0):
        raise ValueError("rain_mm must be a non-empty list of values >= 0")
    if not 0 <= ia_ratio <= 0.5 or baseflow < 0:
        raise ValueError("ia_ratio must be 0-0.5 and baseflow >= 0")

    cn_used = adjust_cn(cn, condition)
    s = retention_mm(cn_used)
    cum_p = np.concatenate(([0.0], np.cumsum(rain)))
    runoff = np.diff(cumulative_runoff_mm(cum_p, cn_used, ia_ratio))
    u, tp, tb = unit_hydrograph(area_km2, tc_hours, dt_hours)
    direct = np.convolve(runoff, u)
    discharge = direct + baseflow
    time = np.arange(direct.size) * dt_hours

    depth = float(runoff.sum())
    total = float(rain.sum())
    warnings = []
    if dt_hours > 0.133 * tc_hours:
        warnings.append("Time step is coarse for this catchment; "
                        "use dt <= 0.133 * tc for a smoother result.")
    if depth == 0:
        warnings.append("No runoff: the rain stays below the initial abstraction.")
    return HydrographResult(
        dt_h=dt_hours, time_h=time, discharge=discharge, rain_mm=rain,
        runoff_mm=runoff, cn_used=float(cn_used), retention_mm=float(s),
        initial_abstraction_mm=float(ia_ratio * s), rain_total_mm=total,
        runoff_depth_mm=depth, runoff_volume_m3=depth * area_km2 * 1000.0,
        runoff_coefficient=depth / total if total > 0 else 0.0,
        peak_m3s=float(discharge.max()),
        time_to_peak_h=float(time[discharge.argmax()]) if depth > 0 else None,
        lag_h=0.6 * tc_hours, tp_h=tp, base_time_h=tb,
        unit_peak=0.208 * area_km2 / tp, warnings=tuple(warnings))


def compare(a: HydrographResult, b: HydrographResult) -> dict:
    """How scenario b differs from scenario a (for example after urbanization)."""
    def pct(x, y):
        return None if x == 0 else 100.0 * (y - x) / x
    shift = (None if a.time_to_peak_h is None or b.time_to_peak_h is None
             else b.time_to_peak_h - a.time_to_peak_h)
    return {
        "peak_change_pct": pct(a.peak_m3s, b.peak_m3s),
        "volume_change_pct": pct(a.runoff_volume_m3, b.runoff_volume_m3),
        "time_to_peak_shift_h": shift,
    }
