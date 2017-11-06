# YamaX
[![license](https://img.shields.io/github/license/Y-modify/YamaX.svg)](LICENSE)

A humanoid made for our research about the automation of walk. See: [YamaX](https://www.y-modify.org/yamax)

# Get started
```shell
docker-compose up -d
ssh -Y root@localhost -p 22000
```

## Setup
In the container,
```shell
. /opt/ros/lunar/setup.sh
catkin_make
. devel/setup.sh
```

### Launch a simurator and controllers
```shell
roslaunch yamax_gazebo world.launch
```
Now you can control joints by publishing to `/yamax/{joint_name}_position_controller/command`
###### Example:
```shell
rostopic pub /yamax/neck_position_controller/command std_msgs/Float64 1.57
```
Will turn the head to 90°

### Launch controllers
```shell
roslaunch yamax_control yamax_control.launch
```

### Rviz
You need `urdf_tutorial` package if you don't have rviz config file(.rviz)
```shell
roslaunch yamax_description display.launch
```
