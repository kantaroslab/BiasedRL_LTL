cd ../catkin_ws
catkin_make
source devel/setup.bash
export TURTLEBOT3_MODEL=waffle_pi
cd src/turtlebot3
roslaunch ./turtlebot3_gazebo/launch/turtlebot3_house.launch

