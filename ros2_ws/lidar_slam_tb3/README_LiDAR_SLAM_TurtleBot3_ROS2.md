# 🧭 Dynamic Navigation & LiDAR-Based SLAM with TurtleBot3 (ROS 2 Jazzy)

This project implements **Dynamic Autonomous Navigation** and **LiDAR-based SLAM** on a simulated TurtleBot3 mobile robot using **ROS 2 Jazzy**, **Gazebo**, **Nav2**, and **slam_toolbox**.

It demonstrates:
- 2D SLAM Mapping using LiDAR  
- Map saving & reuse  
- AMCL-based localization  
- Global path planning  
- Local obstacle avoidance  
- Real-time dynamic navigation in changing environments  

This project is suitable for academic work, robotics research, and portfolio presentation.

---

## 📌 Features

- ✅ **SLAM using LiDAR**  
- ✅ **Real-time map building**  
- ✅ **Accurate localization with AMCL**  
- ✅ **Global & local planners (Nav2)**  
- ✅ **Dynamic obstacle avoidance**  
- ✅ **Fully autonomous navigation to any goal**  
- ✅ Works entirely in simulation — no hardware required

---

## 🛠️ Technologies Used

| Component | Version / Notes |
|----------|-----------------|
| **ROS 2** | Jazzy Jalisco |
| **Gazebo** | TurtleBot3 World |
| **Nav2** | Navigation2 stack |
| **SLAM** | slam_toolbox |
| **LiDAR** | TurtleBot3 LDS-01 (simulated) |
| **Visualization** | RViz2 |

---

# 📍 1. Setup

### Install TurtleBot3 simulation

```bash
sudo apt install ros-jazzy-turtlebot3-gazebo
```

Export TB3 model:

```bash
echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
source ~/.bashrc
```

### Install SLAM Toolbox

```bash
sudo apt install ros-jazzy-slam-toolbox
```

### Install Nav2

```bash
sudo apt install ros-jazzy-nav2-bringup
sudo apt install ros-jazzy-turtlebot3-navigation2
```

---

# 🗺️ 2. Running LiDAR SLAM

### Terminal 1 – Start Gazebo Simulation

```bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

### Terminal 2 – Start SLAM

```bash
ros2 launch slam_toolbox online_async_launch.py
```

### Terminal 3 – RViz2

```bash
LIBGL_ALWAYS_SOFTWARE=1 rviz2
```

Set `Fixed Frame` → **map**

### Terminal 4 – Teleoperation

```bash
ros2 run turtlebot3_teleop teleop_keyboard
```

Drive the robot around to generate a complete map.

---

# 💾 3. Saving the Map

```bash
mkdir -p ~/maps
ros2 run nav2_map_server map_saver_cli -f ~/maps/tb3_world_map
```

This generates:

- `tb3_world_map.pgm`  
- `tb3_world_map.yaml`

---

# 🧭 4. Dynamic Navigation with Nav2

### Terminal 1 – Gazebo

```bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

### Terminal 2 – Launch Nav2 with the saved map

```bash
ros2 launch nav2_bringup bringup_launch.py use_sim_time:=True map:=/home/YOUR_USER/maps/tb3_world_map.yaml
```

### Terminal 3 – RViz2

```bash
LIBGL_ALWAYS_SOFTWARE=1 rviz2
```

Load Nav2 config:

```
/opt/ros/jazzy/share/nav2_bringup/rviz/nav2_default_view.rviz
```

### In RViz:

1. Click **2D Pose Estimate** → set initial pose  
2. Click **2D Goal Pose** → robot navigates autonomously  
3. Insert obstacles in Gazebo → robot avoids them in real time  

---

# 🚀 5. Dynamic Obstacle Avoidance

In Gazebo:

- Open `Insert`
- Add a box, cylinder, or “person” model  
- Place it in front of the robot

Nav2 will:
- Detect the obstacle via LiDAR  
- Update local costmap  
- Recompute the path  
- Avoid the obstacle  

This demonstrates **dynamic navigation**.

---

# 🧪 6. Experiments

Suggested evaluation metrics:
- Time-to-goal  
- Path length  
- Number of replans  
- Obstacle avoidance success  
- Localization stability (AMCL particles)  

You can record data with:

```bash
ros2 bag record /scan /map /tf /cmd_vel /odom
```

---

# 📂 Project Structure

```
.
├── launch/
├── maps/
├── worlds/
├── rviz/
├── src/
└── README.md
```

---

# 📝 License

MIT License

---

# 👤 Author

Developed by **Panagiota Grosdouli**  
Democritus University of Thrace  
2025
