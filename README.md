# BiasedRL_LTL

## **Mission-driven Exploration for Accelerated Deep Reinforcement Learning with Temporal Logic Task Specifications**

**J. Wang, H. Hasanbeig, K. Tan, Z. Sun, and Y. Kantaros**



## Setup Process

**Please run this repo under Ubuntu environment**

**The experiment is run using both RTX 3080 laptop using Ubuntu 20.04 | Python 3.8 | ROS 1 Noetic**

* Setup cuda-toolkit: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local
* For `nvidia-smi`: `sudo apt install nvidia-cuda-toolkit`
* Setup pytorch: https://pytorch.org/get-started/locally/
* Spot: https://spot.lre.epita.fr/install.html
```
./configure --prefix ~/usr; make; make install;
```
* Download ltl2dstar-0.5.4.tar.gz from: https://www.ltl2dstar.de/

```
sudo su 
wget -q -O - https://www.lrde.epita.fr/repo/debian.gpg | apt-key add -
echo 'deb http://www.lrde.epita.fr/repo/debian/ stable/' >> /etc/apt/sources.list
apt-get update
apt-get install spot libspot-dev spot-doc
```
* Example to check if installation is successful: 
```
echo "F G a" > FGa.ltl
ltl2dstar --ltl2nba=spin:ltl2ba --stutter=no --output-format=dot FGa.ltl FGa.dot
dot -Tpdf FGa.dot > FGa.pdf
```

Some commands to follow:
```
pip3 install networkx==2.3
pip3 install numpy==1.23.4
pip3 install tensorflow==2.10.0
pip3 install gym==0.19.0
pip3 install tqdm
pip3 install ffmpeg
pip3 install pygraphviz
pip3 install xmltodict
pip3 install shapely
pip3 install defusedxml
pip3 install netifaces
pip3 install matplotlib
pip3 install pandas
sudo apt install python3.8
sudo apt-get install ros-noetic-dynamixel-sdk
sudo apt-get install ros-noetic-turtlebot3-msgs
sudo apt-get install ros-noetic-turtlebot3
sudo apt-get install graphviz graphviz-dev
sudo apt install ros-noetic-joy ros-noetic-octomap-ros ros-noetic-mavlink
sudo apt install ros-noetic-octomap-mapping ros-noetic-control-toolbox
sudo apt install python3-vcstool python3-catkin-tools protobuf-compiler libgoogle-glog-dev
sudo apt-get install python3-rosdep python3-wstool ros-noetic-ros libgoogle-glog-dev
```

## HOW TO RUN THIS REPO?

* Go to [README_for_Turtlebot3](turtlebot_dqn)
  
