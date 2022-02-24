## Create and Build the Workspace

```
mkdir -p ~/programmed_behavior_ws/src
git clone https://github.com/rodrigodelazcano/Robot-Learning.git
mv Robot-Learning/ProgrammedBehavior/turtlebot3_behavior_navigation/ ~/programmed_behavior_ws/src/
```

Install dependencies

```
cd ~/programmed_behavior_ws/
rosdep update
rosdep install --from-paths src --ignore-src -r -y

```
Build the workspace

```
colcon build
```

## Run Teleoperation

In one terminal launch the Turtlebot Gazebo world
```
ros2 launch  turtlebot3_gazebo turtlebot3_dqn_stage4.launch.py
```

In another terminal source the workspace and launch the teleoperation node
```
. install/local_setup.bash
ros2 run turtlebot3_behavior_navigation teleop
```
* **w**: move forward at a speed of 0.5 m/s.
* **a**: turn left on the spot with a velocity of 0.5 rad/s.
* **d**: turn right on the spot with a velocity of 0.5 rad/s.

## Programmed Behavior with Lidar Sensor

In one terminal launch the Turtlebot Gazebo world
```
ros2 launch  turtlebot3_gazebo turtlebot3_dqn_stage4.launch.py
```

In another terminal source the workspace and launch the teleoperation node
```
. install/local_setup.bash
ros2 run turtlebot3_behavior_navigation random_navigation
```
