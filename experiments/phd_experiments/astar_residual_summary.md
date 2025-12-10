# Residual Heuristic vs Classic – Summary

| Grid type | Method | #Inst | Expansions (mean±std) | Runtime [s] (mean±std) | Path cost (mean±std) |
|-----------|--------|-------|------------------------|------------------------|----------------------|
| maze | classic | 22 | 315.5 ± 75.1 | 0.0012 ± 0.0004 | 63.55 ± 2.25 |
| maze | residual_learned | 22 | 248.7 ± 86.6 | 0.0392 ± 0.0116 | 64.18 ± 2.17 |
| open | classic | 48 | 814.5 ± 50.8 | 0.0032 ± 0.0006 | 62.00 ± 0.00 |
| open | residual_learned | 48 | 181.5 ± 30.7 | 0.0369 ± 0.0075 | 62.00 ± 0.00 |
| rooms | classic | 50 | 846.0 ± 0.0 | 0.0039 ± 0.0020 | 62.00 ± 0.00 |
| rooms | residual_learned | 50 | 130.0 ± 0.0 | 0.0308 ± 0.0117 | 62.00 ± 0.00 |