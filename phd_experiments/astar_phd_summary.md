# A* Learned vs Classic – Summary


| Grid type | Method | #Instances | Expansions (mean±std) | Runtime [s] (mean±std) | Path cost (mean±std) |
|-----------|--------|------------|------------------------|------------------------|----------------------|
| open | classic | 35 | 808.6 ± 52.1 | 0.0035 ± 0.0008 | 62.00 ± 0.00 |
| open | learned | 35 | 310.1 ± 29.8 | 0.0520 ± 0.0083 | 62.00 ± 0.00 |
| open | online_TD | 34 | 807.0 ± 52.0 | 0.0060 ± 0.0012 | 62.00 ± 0.00 |
| open | uncertainty_k0 | 35 | 67.7 ± 4.0 | 0.0159 ± 0.0026 | 65.03 ± 2.26 |
| open | uncertainty_k1 | 34 | 67.7 ± 4.1 | 0.0155 ± 0.0019 | 65.06 ± 2.29 |
| open | uncertainty_k2 | 34 | 67.7 ± 4.1 | 0.0161 ± 0.0026 | 65.06 ± 2.29 |
| open | uncertainty_k3 | 34 | 67.7 ± 4.1 | 0.0165 ± 0.0041 | 65.06 ± 2.29 |