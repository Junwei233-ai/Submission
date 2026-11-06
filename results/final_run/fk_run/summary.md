# B-FK: learned forward kinematics, no gravity term

Fresh test: B-FK 288/300 (93.1–97.7%), expert 293/300 (95.3–98.9%); B-FK vs expert 6/11, p = 0.33; B-FK vs B (URDF) 2/10, p = 0.039.
Validation: B-FK 200/208 (92.6–98.0%).
Compute per 1 kHz step (fresh test, mean): B-FK 342 µs.

## Memory study (learned FK for every level)

| Level | Split | Success | vs L3 (only level / only L3) | never done | restarts |
|---|---|---|---|---|---|
| L1 | validation | 168/208 (74.9–85.5%) | 3/35, p = 6.7e-08 | 13 | 0 |
| L2 | validation | 43/208 (15.7–26.7%) | 0/157, p = 1.1e-47 | 92 | 97 |
| L3 | validation | 200/208 (92.6–98.0%) | – | 1 | 0 |
| L1 | fresh | 239/300 (74.8–83.8%) | 3/52, p = 1.5e-12 | 13 | 0 |
| L2 | fresh | 86/300 (23.8–34.0%) | 0/202, p = 3.1e-61 | 106 | 148 |
| L3 | fresh | 288/300 (93.1–97.7%) | – | 1 | 0 |

## Robustness (60 fresh scenes)

| Condition | B-FK | Expert | B-FK vs expert |
|---|---|---|---|
| noise003 | 58 | 60 | 0/2, p = 0.5 |
| noise010 | 58 | 60 | 0/2, p = 0.5 |
| torque080 | 58 | 60 | 0/2, p = 0.5 |
| pushee30 | 58 | 60 | 0/2, p = 0.5 |
| pushbase20 | 53 | 38 | 18/3, p = 0.0015 |

## Out of distribution (40 scenes each)

| Condition | B-FK | Expert | B-FK vs expert |
|---|---|---|---|
| far_target | 40 | 40 | 0/0, p = 1 |
| high_target | 17 | 15 | 5/3, p = 0.73 |
| low_target | 3 | 30 | 0/27, p = 1.5e-08 |
| wide_start | 29 | 39 | 1/11, p = 0.0063 |

## Holding for 2 s after success (fresh test)

B-FK: 277 of 288 successes stay within 1 cm; expert: 286 of 293; success and held, B-FK vs expert 7/16, p = 0.093.

Forward-kinematics MLP: 268,291 parameters; validation final-frame error median 2.0 mm, p95 6.2 mm, max 15.4 mm.
