# Ground Robot Setup & Run 

To get your own dataset:

* Collect data: `python3 ros_data_collect.py`
* Train BiasedNN: `python3 nntrain_softmax.py`

To run the ground robot (Turtlebot3 Waffle Pi), please run the code accordingly

* Train RL Agent: `python3 train_ros.py`

Lab experiment setup:
```
Current setup (detail)
The chair (obstacle), has radius of 0.35 meter

3 goals (red dots)
Goal 1: (0.4, 0.4) | radius: 8 cm
Goal 2: (-0.4, -0.4) | radius: 8 cm
Goal 3: (-1, 1) | radius: 8 cm

4 obstacles (cylinder)
Obs 1: (0.35 -0.35) | radius: 25 cm
Obs 2: (-1, -1) | radius: 30 cm
Obs 3: (-0.35 0.35) | radius: 25 cm
Obs 4: (1, 1) | radius: 3 cm

high variance area: 
-0.4 m <= x, y <= 0.4 m

the path that connects goal 1 and goal 2 (where the robot is currently located) will have very high variance, which means theoretically the car might have a high chance to crash into the obstacles if you want to go directly from goal 1 to goal 2 via the straight path
```

## Notes

To setup Turtlebot3 in the lab, please check this photo

![](https://emanual.robotis.com/assets/images/platform/turtlebot3/software/network_configuration.png)

IP_OF_REMOTE_PC: the IPv4 address of your computer that starts the `roscore`

IP_OF_TURTLEBOT: the IPv4 address of the turtlebot 


Check the `~/.bashrc` file both in RasberryPi and your own device

https://github.com/ROBOTIS-GIT/turtlebot3/issues/510 

#### Data Collection with Risk Aware

* networkx grid

```python
G = nx.grid_2d_graph(goal_nums, goal_nums)  # 5x5 grid
G.remove_nodes_from([(0,0)])
G[(0, 0)][(0, 1)]['weight'] = 3
nx.set_edge_attributes(G, values=1, name='weight')
print(type(G), nx.is_weighted(G), "\n", G.edges(), "\n", G.nodes())
```
```commandline
1 Line Solution is y = 1.125x + 0.03
2 Line Solution is y = 1.1249999999999998x + -0.12999999999999995
```

## Experiment RUNTIME (Minutes)

### Task 1: <>goal_1 && <>goal_2 && <>goal_3 && []!obstacles

#### 3 Obstacle

```
Biased NN Training Time: 482 minutes

Our Method:

Test 1: 890
Test 2: 867
Test 3: 807

Average Time ~ 855 minutes

Train Time for DQN = 855 + 482 = 1337
```

#### 10 Obstacle

```
Biased NN Training Time: 625 minutes

Our Method:

Test 1: 1161
Test 2: 1014
Test 3: 1097

Average Time ~ 1090 minutes

Train Time for DQN = 1090 + 625 = 1715
```

### Task 4: <>goal_1 && <>goal_2 && (!goal_2 U goal_1) && <>goal_3 && []!obstacles

```
Biased NN Training Time: 236 minutes

Our Method:
Test 1: 1700
Test 2: 1612
Test 3: 1632

Average Time ~ 1648 minutes

Train Time for DQN = 1648 + 236 = 1884 
```
