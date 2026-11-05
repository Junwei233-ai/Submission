# Memory study

| Level | Split | Success | 95% CI | vs B (only level / only B, p) | never done | restarts | arm drift before done, median (rad) | creep after done, median (cm) |
|---|---|---|---|---|---|---|---|---|
| L1 Four single-frame MLPs + latch | validation | 189/208 | 86.2-94.1% | 1/19, p=4e-05 | 11 | 0 | 0.057 | 0.7 |
| L2 GRU, no stop head | validation | 54/208 | 20.5-32.3% | 0/153, p=1.8e-46 | 58 | 122 | 0.314 | 9.8 |
| L3 B: GRU + stop head + latch | validation | 207/208 | 97.3-99.9% | -- | 1 | 0 | 0.038 | 0.4 |
| L1 Four single-frame MLPs + latch | fresh | 272/300 | 86.8-93.5% | 3/27, p=8.4e-06 | 12 | 0 | 0.057 | 0.7 |
| L2 GRU, no stop head | fresh | 83/300 | 22.9-33.0% | 0/213, p=1.5e-64 | 78 | 188 | 0.316 | 13.8 |
| L3 B: GRU + stop head + latch | fresh | 296/300 | 96.6-99.5% | -- | 2 | 0 | 0.038 | 0.4 |

Direct-torque ablation (validation): 0/208, median closest approach 13.9 cm

Gains: L1 100/30/0.2, L2 200/30/0.2, L3 200/30/0.2
