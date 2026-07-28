# Benchmark results

## Scenarios

- **regulation** - Recover from an initial 11.5 deg tilt and hold the cart at x = 0.
- **disturbance** - Balanced upright, then hit with a 12 N impulse push for 50 ms at t = 1 s.
- **tracking** - Move the cart 1 m to the right at t = 0.5 s without dropping the pole.
- **noisy-sensors** - Regulation with 0.5 deg angle noise, 2 mm encoder noise and noisy rates.
- **swing-up** - Start hanging down at 180 deg, pump energy, then catch and balance.

## Scores

| Scenario | Controller | Success | Settling [s] | Peak angle [deg] | RMS angle [deg] | Peak force [N] | Effort [N^2 s] | ITAE |
|---|---|:---:|---:|---:|---:|---:|---:|---:|
| regulation | PID | yes | 2.26 | 11.46 | 1.56 | 8.72 | 9.54 | 0.062 |
| regulation | LQR | yes | 1.97 | 11.46 | 1.77 | 10.00 | 3.04 | 0.055 |
| regulation | MPC | yes | 1.98 | 11.46 | 1.81 | 10.00 | 2.78 | 0.057 |
| disturbance | PID | yes | 3.28 | 8.90 | 1.47 | 10.00 | 20.48 | 0.096 |
| disturbance | LQR | yes | 2.13 | 3.90 | 0.69 | 10.00 | 5.57 | 0.048 |
| disturbance | MPC | yes | 2.32 | 4.25 | 0.78 | 10.00 | 5.40 | 0.055 |
| tracking | PID | yes | 3.11 | 10.30 | 1.81 | 10.00 | 16.91 | 0.199 |
| tracking | LQR | yes | 2.71 | 8.00 | 2.30 | 8.66 | 2.43 | 0.229 |
| tracking | MPC | yes | 2.71 | 7.97 | 2.30 | 7.25 | 2.11 | 0.230 |
| noisy-sensors | PID | yes | 2.37 | 8.59 | 1.13 | 6.29 | 14.55 | 0.109 |
| noisy-sensors | LQR | yes | 2.09 | 8.59 | 1.22 | 8.41 | 8.04 | 0.072 |
| noisy-sensors | MPC | yes | 2.11 | 8.59 | 1.25 | 7.27 | 6.05 | 0.073 |
| swing-up | SwingUp+LQR | yes | 5.12 | 7.83 | 2.52 | 10.00 | 49.18 | 5.356 |

## Design report

- Open-loop poles: `-5.74+0.00j, -0.14+0.00j, +0.00+0.00j, +5.44+0.00j` - one in the right half plane, so the plant is unstable.
- LQR gain `K = [-8.66, -10.33, -59.09, -11.18]`
- Closed-loop poles: `-24.26+0.00j, -5.47+0.00j, -1.38+1.00j, -1.38-1.00j`
