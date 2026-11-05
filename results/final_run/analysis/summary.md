# Additional analyses of variant B (frozen model unchanged)

## Training seeds (validation, chosen gains)

| GRU seed | 1 cm success (95% CI) | best epoch | val arm loss | latch |
|---|---|---|---|---|
| 42 (frozen) | 207/208 (97–100%) | 78 | 0.1633 | 1 |
| 1 | 207/208 (97–100%) | 52 | 0.1712 | 2 |
| 2 | 208/208 (98–100%) | 78 | 0.1713 | 3 |

## Learning curve (validation, chosen gains; park/stop MLPs and GRU retrained on nested subsets)

| fraction | train trajectories | 1 cm success (95% CI) | val wheel loss | val arm loss | val stop loss | train time (min) |
|---|---|---|---|---|---|---|
| 0.1 | 97 | 157/208 (69–81%) | 0.283 | 0.8584 | 0.1621 | 6 |
| 0.25 | 243 | 193/208 (88–96%) | 0.1559 | 0.465 | 0.1084 | 11 |
| 0.5 | 486 | 195/208 (90–96%) | 0.08638 | 0.2491 | 0.04241 | 19 |
| 1.0 | 973 | 207/208 (97–100%) | 0.05367 | 0.1633 | 0.02145 | 31 |

## Gravity term of the PD layer (validation, paired)

| gains | with gravity | without | discordant with/without (p) | median final error without |
|---|---|---|---|---|
| kp200_kd30_lead0.2 | 207/208 (97–100%) | 207/208 (97–100%) | 0/0 (1) | 0.0100 m |
| kp50_kd30_lead0.1 | 203/208 (94–99%) | 57/208 (22–34%) | 147/1 (8.4e-43) | 0.3187 m |

## Holding after success (fresh test, 2 s more simulation)

| controller | 1 cm success | held within 1 cm (whole 2 s) | held within 2 cm | within 1 cm over the last 1 s | worst error in the last 1 s: median / p95 | error at the end: median / p95 |
|---|---|---|---|---|---|---|
| B | 296/300 | 286 | 296 | 296 | 2.2 / 3.6 mm | 0.6 / 2.1 mm |
| v3 | 289/300 | 171 | 258 | 228 | 5.7 / 19.0 mm | 2.5 / 8.1 mm |
| expert | 293/300 | 286 | 293 | 293 | 2.4 / 3.4 mm | 0.6 / 2.1 mm |
| diag_v3_kp100_kd30_lead0.1 | 289/300 | 278 | 289 | 289 | 2.3 / 4.1 mm | 0.7 / 2.1 mm |
| diag_B_kp100_kd10_lead0.1 | 296/300 | 144 | 254 | 217 | 6.1 / 23.9 mm | 2.3 / 8.1 mm |

Success and held within 1 cm for the whole 2 s, B vs expert: discordant 8/8, exact McNemar p = 1.
Success and held within 1 cm for the whole 2 s, B vs v3: discordant 122/7, exact McNemar p = 3.1e-28.
