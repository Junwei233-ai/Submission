# Evaluation suite (main: B, baseline: v3, reference: expert)

1 cm successes (rate, Wilson 95% CI). McNemar: exact two-sided, discordant pairs shown as first/second.

| condition | n | B | v3 | expert | B vs expert (p) | B vs v3 (p) | median err B / expert (m) |
|---|---|---|---|---|---|---|---|
| fresh | 300 | 296 (98.7%, 97–99) | 289 (96.3%, 94–98) | 293 (97.7%, 95–99) | 7/4 (0.55) | 10/3 (0.092) | 0.0100 / 0.0100 |
| ood_far_target | 40 | 40 (100.0%, 91–100) | - | 40 (100.0%, 91–100) | 0/0 (1) | - | 0.0100 / 0.0100 |
| ood_high_target | 40 | 40 (100.0%, 91–100) | - | 15 (37.5%, 24–53) | 25/0 (6e-08) | - | 0.0100 / 0.0175 |
| ood_low_target | 40 | 29 (72.5%, 57–84) | - | 30 (75.0%, 60–86) | 7/8 (1) | - | 0.0100 / 0.0100 |
| ood_wide_start | 40 | 38 (95.0%, 83–99) | - | 39 (97.5%, 87–100) | 1/2 (1) | - | 0.0100 / 0.0100 |
| robust_noise003 | 60 | 59 (98.3%, 91–100) | - | 60 (100.0%, 94–100) | 0/1 (1) | - | 0.0100 / 0.0100 |
| robust_noise010 | 60 | 59 (98.3%, 91–100) | - | 60 (100.0%, 94–100) | 0/1 (1) | - | 0.0100 / 0.0100 |
| robust_pushbase20 | 60 | 56 (93.3%, 84–97) | - | 38 (63.3%, 51–74) | 19/1 (4e-05) | - | 0.0100 / 0.0100 |
| robust_pushee30 | 60 | 59 (98.3%, 91–100) | - | 60 (100.0%, 94–100) | 0/1 (1) | - | 0.0100 / 0.0100 |
| robust_torque080 | 60 | 59 (98.3%, 91–100) | - | 60 (100.0%, 94–100) | 0/1 (1) | - | 0.0100 / 0.0100 |

Controller compute per 1 kHz step on the fresh test (features, GRU update every 10th step, PD + gravity; expert: IK + PD + gravity): B 87 µs, v3 82 µs, expert 82 µs.
Model B: 616 KiB, GRU 60186 parameters + park MLP 17540.
Model v3: 477 KiB, GRU 60186 parameters.
