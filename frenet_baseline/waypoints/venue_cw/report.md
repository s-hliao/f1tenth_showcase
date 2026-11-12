# Racelines: venue_cw

Generated 2026-09-29T09:17:51, git unknown

Direction: cw. Clearance required: car half-width 0.155 m + gap 0.1 m = 0.255 m.

| line | length m | tightest radius m | min wall clearance m | min speed | max speed | lap s | checks |
|---|---|---|---|---|---|---|---|
| stable | 165.2 | 0.85 | 0.11 | 1.62 | 7.97 | 37.75 | FAIL |
| optimal | 164.0 | 0.94 | 0.25 | 1.70 | 7.94 | 37.11 | FAIL |
| fast | 157.0 | 1.17 | 0.25 | 1.90 | 7.54 | 34.19 | FAIL |

Fast line solver status: Solve_Succeeded

## Limits used

- a_y: a_y_max_used=3.1 (vehicle_params.yaml)
- a_brake: a_x_max_used=3.1 (vehicle_params.yaml)
- a_accel: a_max=2.4 (vehicle_params.yaml)
- kappa_max: tan(steer_max)/L = 0.82 (vehicle_params.yaml)
- v_max: 8.0 (--v-max)

Lap times are estimates from the grip-limited speed profile, not simulation results.
