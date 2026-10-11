**[Open the interactive app](https://biswashk920.github.io/hydrograph-lab/)**

# Hydrograph lab

Turn a storm into a flood hydrograph. Rainfall losses use the SCS Curve Number
method and the catchment response uses the SCS triangular unit hydrograph.
Python package plus an interactive page in `docs/` (works on GitHub Pages).

**Inputs:** catchment area, curve number, soil moisture (dry, normal, wet), time of
concentration (with a Kirpich estimate), storm depth, duration and shape (uniform,
front-loaded, middle-peaked, back-loaded, SCS Type II 24 h, or your own values),
time step, initial loss ratio, baseflow.
**Outputs:** runoff depth and volume, runoff coefficient, peak flow, time to peak,
lag time and the hydrograph. A compare mode overlays a second scenario, for example
the same catchment after urbanization.

## Method
1. **Losses.** `S = 25400/CN - 254` (mm), `Ia = ratio * S`,
   `Q = (P - Ia)^2 / (P - Ia + S)` for `P > Ia`, applied to cumulative rainfall.
   With ratio 0.2 this is the textbook `(P - 0.2S)^2 / (P + 0.8S)`.
2. **Unit hydrograph.** lag = 0.6 tc, Tp = dt/2 + lag, Qp = 0.208 A / Tp
   (m3/s per mm), base time 2.67 Tp. Ordinates are scaled slightly so the volume is
   exactly A x 1000 m3 per mm of runoff.
3. **Hydrograph.** Convolution of each step's runoff with the unit hydrograph,
   plus baseflow.

Units: mm, km2, hours, m3/s.

## Limits
The catchment is treated as one lump, the response is assumed linear, the triangle
is an approximation, and the curve number is constant during the storm. Use
`dt <= 0.133 tc` for a smooth result. The SCS Type II storm uses tabulated points
interpolated linearly; check them against the NRCS table before relying on them.

## Checks
`python -m pytest` runs 14 tests, including a hand-calculated example
(100 mm on CN 75 gives 41.1 mm of runoff), a mass balance (runoff volume equals
runoff depth times area, and the hydrograph integrates to the same volume), and
the effect of urbanization (higher peak, earlier peak).

## Run it
```
pip install -r requirements.txt
python -m pytest
python -m http.server 8000 --directory docs
```
Then open http://localhost:8000.

```python
from hydrograph_lab import run, storm, compare

rain = storm("middle", depth_mm=100, duration_h=6, dt_h=0.1)
before = run(area_km2=25, cn=65, tc_hours=2.5, rain_mm=rain)
after = run(area_km2=25, cn=90, tc_hours=1.2, rain_mm=rain)
print(before.peak_m3s, after.peak_m3s, compare(before, after))
```

## References
- USDA NRCS, National Engineering Handbook, Part 630 Hydrology (chapters 9, 10 and 16).
- Chow, Maidment and Mays, Applied Hydrology.
