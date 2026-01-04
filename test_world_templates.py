import numpy as np
from world_templates import world_bottleneck, world_culdesac, world_noisy_corridor

I = np.zeros((50, 50))

Ib, meta_b = world_bottleneck(I)
Ic, meta_c = world_culdesac(I)
In, meta_n = world_noisy_corridor(I)

print(meta_b)
print(meta_c)
print(meta_n)
