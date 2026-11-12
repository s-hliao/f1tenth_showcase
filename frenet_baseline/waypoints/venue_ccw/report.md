# Racelines: venue_ccw

Generated 2026-09-29T09:18:58, git unknown

Direction: ccw. Clearance required: car half-width 0.155 m + gap 0.1 m = 0.255 m.

| line | length m | tightest radius m | min wall clearance m | min speed | max speed | lap s | checks |
|---|---|---|---|---|---|---|---|
| stable | 165.4 | 0.99 | 0.35 | 1.75 | 7.94 | 38.01 | FAIL |
| optimal | 163.4 | 0.92 | 0.07 | 1.69 | 7.90 | 36.71 | FAIL |
| fast | 157.1 | 1.14 | 0.25 | 1.88 | 7.61 | 34.21 | FAIL |

Fast line solver status: Solve_Succeeded

## Limits used

- a_y: a_y_max_used=3.1 (vehicle_params.yaml)
- a_brake: a_x_max_used=3.1 (vehicle_params.yaml)
- a_accel: a_max=2.4 (vehicle_params.yaml)
- kappa_max: tan(steer_max)/L = 0.82 (vehicle_params.yaml)
- v_max: 8.0 (--v-max)

Lap times are estimates from the grip-limited speed profile, not simulation results.
