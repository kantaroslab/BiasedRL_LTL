"""
For Turtlebot3 Waffle_pi usage only
"""
from shapely.geometry import Polygon
import numpy as np
import math
import rospy
import logging
from sensor_msgs.msg import LaserScan
from gazebo_msgs.srv import *
import tf
import random
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from SAC.tools import all_collision_inspect


def publish_action_to_gazebo(state, action, is_noisy=False, set_lab_noise=False):
    noise_v, noise_w = 0, 0
    if is_noisy:
        if set_lab_noise:
            """
            y1 = 1.125x1 + 0.03
            y2 = 1.125x2 + -0.13
            """
            x, y, _ = state
            noise_v = np.random.normal(0.001, 0.001)
            if -0.3 < x < 0.3 and -0.3 < y < 0.3:
                # apply high covariance noise in target area
                noise_w = np.random.normal(0.002, 0.5)
            else:
                noise_w = np.random.normal(0.001, 0.001)
        else:
            noise_v = np.random.uniform(-0.002, 0.002)
            noise_w = np.random.uniform(-0.002, 0.002)
    linear_velocity, angular_velocity = action
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    vel = Twist()
    vel.linear.x = clamp(linear_velocity + noise_v, -0.26, 0.26)
    vel.angular.z = clamp(angular_velocity + noise_w, -1.82, 1.82)
    pub.publish(vel)


def reset_gazebo_with_designed_state(state):
    x, y, angle = state[0], state[1], state[2]
    rospy.wait_for_service('/gazebo/set_model_state')
    model_state = ModelState()
    model_state.model_name = 'turtlebot3_waffle_pi'
    model_state.pose.position.x = x
    model_state.pose.position.y = y
    quaternion = tf.transformations.quaternion_from_euler(0, 0, angle)
    model_state.pose.orientation.x = quaternion[0]
    model_state.pose.orientation.y = quaternion[1]
    model_state.pose.orientation.z = quaternion[2]
    model_state.pose.orientation.w = quaternion[3]
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    set_state(model_state)
    publish_action_to_gazebo(state, [0, 0])
    rospy.sleep(0.5)


def reset_turtlebot_state(ws_size):
    feasible_ws = ws_size - 0.26
    init_x = random.uniform(-feasible_ws, feasible_ws)
    init_y = random.uniform(-feasible_ws, feasible_ws)
    init_theta = random.uniform(-np.pi, np.pi)
    state = [init_x, init_y, init_theta]
    return state


def reset_obs_free_risk_aware(obstacles, ws_size, agent_size):
    """
    reset the state in risk_aware_data_collection with a more conservative value
    """
    s = reset_turtlebot_state(ws_size)
    for elm in obstacles:
        while all_collision_inspect(elm, s[0], s[1], agent_size):
            s = reset_turtlebot_state(ws_size)
    return s

def reset_obs_free(obstacles, ws_size):
    s = reset_turtlebot_state(ws_size)
    for elm in obstacles:
        while all_collision_inspect(elm, s[0], s[1]):
            s = reset_turtlebot_state(ws_size)
    return s


def get_laser_reading():
    msg = rospy.wait_for_message("scan", LaserScan)
    laser_result = [item for item in list(msg.ranges) if item < float('inf')]
    min_laser_dis = min(laser_result)  # The distance to judge if the robot is hitting obstacles
    return min_laser_dis


def init_obstacles_check_gazebo():
    # Check if the initial position of the robot hits the obstacles (using laserscan)
    min_laser_distance = get_laser_reading()
    # logging.info("Current min laser distance: {}".format(min_laser_distance))
    if min_laser_distance < 0.14:
        return True
    return False


def get_cur_gazebo_model_state():
    get_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    ros_model = GetModelStateRequest()
    ros_model.model_name = 'turtlebot3_waffle_pi'
    ros_model_state = get_state_service(ros_model)
    (_, _, yaw) = tf.transformations.euler_from_quaternion([ros_model_state.pose.orientation.x,
                                                            ros_model_state.pose.orientation.y,
                                                            ros_model_state.pose.orientation.z,
                                                            ros_model_state.pose.orientation.w])
    state = [ros_model_state.pose.position.x, ros_model_state.pose.position.y, yaw]
    return state


def check_sys2ap_rabin_ros(state, ap_range_map):
    x, y, _ = state
    result_ap = None
    is_hitting_obs = False
    for ap, elements in ap_range_map.items():
        if ap != 'obstacles':
            if all_collision_inspect(elements, x, y):
                result_ap = ap
        else:
            for val in elements:
                if all_collision_inspect(val, x, y):
                    result_ap = ap
                    is_hitting_obs = True
    if is_hitting_obs:
        """
        If both non-obstacle AP and 'obstacles' are satisfied
        Then the obstacles will be satisfied prior to the goal
        But one should also check the position of the target
        """
        result_ap = "obstacles"
    return result_ap


def check_sys2ap_rabin(state, env, ap_range_map):
    x1, y1, _ = state
    assert env.chopper.icon_w == env.chopper.icon_h
    agent_size = env.chopper.icon_w
    for k, v in ap_range_map.items():
        x2, y2, size2_w, size2_h = v
        assert size2_w == size2_h
        if collision_inspect(x1, y1, agent_size, x2, y2, size2_w):
            # If initial state of this episode is not in any of AP
            # Then it is guaranteed to be in rabin state 1
            return k
    return None


def collision_inspect(x1, y1, s1_size, x2, y2, s2_size):
    # Calculate collision under the assumption that
    # size & coordinate of the goal & obstacles are known to us
    poly1 = Polygon([(x1, y1),
                     (x1 + s1_size, y1),
                     (x1 + s1_size, y1 + s1_size),
                     (x1, y1 + s1_size)])
    poly2 = Polygon([(x2, y2),
                     (x2 + s2_size, y2),
                     (x2 + s2_size, y2 + s2_size),
                     (x2, y2 + s2_size)])
    return poly1.intersects(poly2)


def move_real_with_action_out(xv, uv, ts, ws_size, take_noise=False, set_lab_noise=False):
    if take_noise:
        if set_lab_noise:
            x, y, _ = xv
            noise_v = np.random.normal(0.001, 0.001)

            if -0.3 < x < 0.3 and -0.3 < y < 0.3:
                noise_w = np.random.normal(0.002, 0.5)
            else:
                noise_w = np.random.normal(0.001, 0.001)
            uv0 = clamp(uv[0] + noise_v, -0.26, 0.26)
            uv1 = clamp(uv[1] + noise_w, -1.82, 1.82)
            uv = [uv0, uv1]
        else:
            # other experiments
            process_noise = (
                    0.002 * np.dstack(
                [-np.ones(2), np.ones(2)]
            )[0]
            )
            uv_ = np.array(uv)
            noise = np.random.uniform(
                low=process_noise[:, 0],
                high=process_noise[:, 1],
                size=uv_.shape,
            )
            uv = uv + noise
    state_next = dynamics(xv, uv, ts)
    xp = clamp(state_next[0], -ws_size, ws_size)
    yp = clamp(state_next[1], -ws_size, ws_size)
    tp = clamp(state_next[2], -np.pi, np.pi)
    return [xp, yp, tp], uv


def move_real(xv, uv, ts, ws_size, take_noise=False, set_lab_noise=False):
    # wp_size: workspace_size
    # Add noise on action !!
    if take_noise:
        if set_lab_noise:
            """
            For lab experiment, set high covariance for area near goal_1 and goal_2
            Goal 1: (0.4, 0.4) | radius: 8 cm
            Goal 2: (-0.4, -0.4) | radius: 8 cm

            For area [-0.4, -0.4] ~ [0.4, 0.4]
            N (0.002, 2)
            
            y1 = 1.125x1 + 0.03
            y2 = 1.125x2 + -0.13

            Others
            N (0.002, 0.001)
            """
            x, y, _ = xv
            noise_v = np.random.normal(0.002, 0.001)
            if -0.3 < x < 0.3 and -0.3 < y < 0.3:
                noise_w = np.random.normal(0.002, 0.5)
            else:
                noise_w = np.random.normal(0.002, 0.001)

            # print("({}, {}) | omega_noise: {}".format(x, y, noise_w))
            uv0 = clamp(uv[0] + noise_v, -0.26, 0.26)
            uv1 = clamp(uv[1] + noise_w, -1.82, 1.82)
            uv = [uv0, uv1]
            # if -0.38 < x < 0.38 and -0.38 < y < 0.38:
            #     print("uv: {}".format(uv))
        else:
            # other experiments
            process_noise = (
                    0.002 * np.dstack(
                [-np.ones(2), np.ones(2)]
            )[0]
            )
            uv_ = np.array(uv)
            noise = np.random.uniform(
                low=process_noise[:, 0],
                high=process_noise[:, 1],
                size=uv_.shape,
            )
            uv = uv + noise
    state_next = dynamics(xv, uv, ts)
    xp = clamp(state_next[0], -ws_size, ws_size)
    yp = clamp(state_next[1], -ws_size, ws_size)
    tp = clamp(state_next[2], -np.pi, np.pi)
    return [xp, yp, tp]


def dynamics(xv, uv, ts):
    res = np.zeros_like(xv)
    tmp = uv[1] * ts / 2 + 1e-6
    sinc = np.divide(np.sin(tmp * np.pi), (tmp * np.pi))
    cal = np.multiply(uv[0], sinc)
    tmpx = np.multiply(cal, np.cos(xv[2] + tmp))
    tmpy = np.multiply(cal, np.sin(xv[2] + tmp))
    res[0] = xv[0] + tmpx
    res[1] = xv[1] + tmpy
    res[2] = xv[2] + ts * uv[1]
    return res


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


def region_counter_turtlebot(state, grid_check):
    # count which region the robot is initialized to
    # for turtlebot usage only
    x, y, _ = state  # angle information is not needed
    print(grid_check, len(grid_check))
    exit(0)
