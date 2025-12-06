echo 'import numpy as np
from nav_research.vla.vla_nbv_planner import VLA_NBV_Planner

coverage = np.random.rand(50,50)
uncertainty = np.random.rand(50,50)
features = np.random.rand(50,50)
drift = np.random.rand(50,50)
obstacles = np.zeros((50,50))
obstacles[10:20, 10:40] = 1

planner = VLA_NBV_Planner(coverage, uncertainty, features, drift, obstacles)

command = "explore high uncertainty near walls"
goal, costmap = planner.pick_nbv(command)

print("NBV goal:", goal)' > nav_research/vla/vla_demo.py
