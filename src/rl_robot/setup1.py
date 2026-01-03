entry_points={
    'console_scripts': [
        'basic_controller = rl_robot.basic_controller:main',
        'rl_agent = rl_robot.rl_agent:main',
        'fsm_controller = rl_robot.fsm_controller:main',
        'benchmark = rl_robot.benchmark_node:main',
        'astar_planner = rl_robot.astar_planner:main',
    ],
},
