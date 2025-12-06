# Ablation: Hidden Size vs Performance

| Method | #Instances | Expansions (mean±std) | Runtime [s] (mean±std) | Path cost (mean±std) |
|--------|------------|------------------------|------------------------|----------------------|
| classic | 48 | 814.5 ± 50.8 | 0.0040 ± 0.0013 | 62.00 ± 0.00 |
| learned_h128 | 48 | 65.8 ± 3.0 | 0.0197 ± 0.0063 | 62.67 ± 1.11 |
| learned_h32 | 48 | 65.2 ± 2.3 | 0.0187 ± 0.0049 | 62.79 ± 1.40 |
| learned_h64 | 48 | 70.5 ± 4.5 | 0.0191 ± 0.0056 | 66.88 ± 2.51 |