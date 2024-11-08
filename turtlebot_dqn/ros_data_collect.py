import numpy as np
import os
import pandas as pd
import random
from shapely.geometry import Polygon
import time
import csv
from itertools import product
import logging
from SAC.train_helper import check_folder
# import visilibity as vis
import xmltodict
import math
import tf 
import pickle
import rospy
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import *
from geometry_msgs.msg import Twist
from LTL2DRA.dra_utils import * 
from SAC.tools import all_collision_inspect
import shutil
from config.config import load_config


def load_env():
    config_name = "./config/ground_robot.yaml"
    config = load_config(config_name)
    data_folder = config['data_folder']
    if os.path.exists(data_folder):
        # remove previous directory
        shutil.rmtree(data_folder)
    check_folder(data_folder)
    log_name = os.path.join(data_folder, "data_collection.log")

    level = logging.INFO
    format = '%(message)s'
    handlers = [logging.FileHandler(log_name), logging.StreamHandler()]
    # noinspection PyArgumentList
    logging.basicConfig(level=level, format=format, handlers=handlers)

    file_name = os.path.join(data_folder, "data.csv")
    if os.path.exists(file_name):
        logging.info("Removing current csv file: {}".format(file_name))
        os.remove(file_name)

    # get environment basic info
    logging.info("Loading gazebo world information.")
    # model_sdf = '../catkin_ws/src/turtlebot3/turtlebot3_gazebo/models/turtlebot3_world/model.sdf'
    model_sdf = config['world_model_sdf']
    with open(model_sdf, 'r', encoding='utf-8') as file:
        mysdf = file.read()
    sdf_dict = xmltodict.parse(mysdf)
    sdf_dict = sdf_dict['sdf']['model']['link']
    vis_list = sdf_dict['visual']

    obstacles = []  # goals not included
    goal_size = 0
    tmp = []
    
    for ind in range(len(vis_list)):
        name = vis_list[ind]['@name']
        if name.startswith("goal"):
            goal_size = float(vis_list[ind]['geometry']['cylinder']['radius'])
            m = list(map(float, vis_list[ind]['pose'].split(" ")))
            radius = float(vis_list[ind]['geometry']['cylinder']['radius'])
            tmp = [m[0], m[1], radius]
        elif name.startswith("wall"):
            m = list(map(float, vis_list[ind]['pose'].split(" ")))
            box = list(map(float, vis_list[ind]['geometry']['box']['size'].split(" ")))
            tmp = [m[0], m[1], box[0], box[1]]  # [x, y, width, height]
            obstacles.append(tmp)
        elif name.startswith("obs"):
            if 'cylinder' in vis_list[ind]['geometry'].keys():
                m = list(map(float, vis_list[ind]['pose'].split(" ")))
                radius = float(vis_list[ind]['geometry']['cylinder']['radius'])
                # Add padded radius
                if config['padding_perc'] != 0:
                    logging.info("Padded Radius Injected, increased by {}%".format(config['padding_perc'] * 100))
                radius_padded = radius + radius * config['padding_perc'] 
                tmp = [m[0], m[1], radius_padded]
                obstacles.append(tmp)
            elif 'box' in vis_list[ind]['geometry'].keys():
                m = list(map(float, vis_list[ind]['pose'].split(" ")))
                box = list(map(float, vis_list[ind]['geometry']['box']['size'].split(" ")))
                tmp = [m[0], m[1], box[0], box[1]]  # [x, y, width, height]
                obstacles.append(tmp)
            else:
                continue
        logging.info("{}: {}".format(name, tmp))


    # setup multiple goal list
    workspace_ = config['ws_size']
    goal_nums = config['goal_nums']
    # goal_part_step = config['goal_part_step']
    agent_size = config['agent_size'] 
    workspace = workspace_ - agent_size
    start = -1 * workspace
    stop = workspace 
    # result = np.linspace(start, stop, int((stop - start) / goal_part_step + 1))
    result = np.linspace(start, stop, goal_nums)
        
    goal_list_pre = list(product(result, result))
    goal_list = []
    logging.info("Workspace:{} | nums:{}".format(workspace, len(goal_list_pre)))
    for g in goal_list_pre:
        flag = False
        for elm in obstacles:
            if all_collision_inspect(elm, g[0], g[1]):
                flag = True
                break 
        if not flag:
            goal_list.append(g)

    logging.info("Available goals: {}".format(len(goal_list)))
    for g in goal_list:
        logging.info(g)
    logging.info("\n")
    
    # discretize the action space
    dis_v = list(np.linspace(-config['max_v'], config['max_v'], config['v_discrete']))
    dis_w = list(np.linspace(-config['max_w'], config['max_w'], config['w_discrete']))
    dis_action = list(product(dis_v, dis_w))
    action_map = {}
    logging.info("\nNumber of discrete actions: {}\n".format(len(dis_action)))
    act_num = 0
    for act in dis_action:
        logging.info(act)
        action_map[act] = act_num
        act_num += 1

    
    goal_change = config['data_collect_goal_change']
    i = 0
    ind = 0
    start_time = time.time()
    rerun_flag = False
    cur_goal = None
    while True:
        # change goal every N rounds
        if i % goal_change == 0:
            if ind >= len(goal_list):
                print("Data collection completed, exiting.")
                break
            cur_goal = goal_list[ind]
            ind += 1
            gx, gy = cur_goal[0], cur_goal[1] 
            logging.info("i:{} | Current goal selection is: ({}, {}) | Time: {}".format(i, gx, gy, time.time() - start_time))
        if rerun_flag:
            rerun_flag = False

        state = reset_obs_free(obstacles, workspace)

        # Check if goals and states hits in the first place
        cur_goal_with_radius = list(cur_goal)
        cur_goal_with_radius.append(goal_size)
        tmp_goal = state
        data_goal = []
        if all_collision_inspect(cur_goal_with_radius, state[0], state[1]):
            tmp_goal += [gx, gy, action_index, best_action[0], best_action[1], s_next[0], s_next[1], s_next[2]]
            data_goal.append(tmp_goal)

            if not os.path.exists(file_name):
                pd.DataFrame(data_goal).to_csv(file_name, header=False, index=False)
            else:
                with open(file_name, 'a', newline='') as fd:
                    writer = csv.writer(fd)
                    writer.writerows(data_goal)
            i += 1
            continue

        """
        data structure: 
        [x0, y0, t0, xf, yf, index, action[0], action[1], x1, y1, t1]
        """
        data = []
        accept_action_dict = {}

        act_dis_dict = {}
        run_times = config['data_run_times']
        tmp = state
        # if a certain percentage of points has collied with obstacles, reject them
        failure_tolerance = config['bias_fail_tolerance']
        threshold_marker = {action: 0 for action in dis_action}
        for action in dis_action:
            t = 0
            obs_threshold_cnt = 0
            while t < run_times:
                # Do not break the loop to avoid all empty set
                t += 1
                s_next = move_real(state, action, ts=config['sample_rate'], ws_size=workspace, take_noise=True, set_lab_noise=config['use_lab_noise_setup'])
                flag = False
                for elm in obstacles:
                    if all_collision_inspect(elm, s_next[0], s_next[1], agent_size):
                        flag = True 
                        break
                if flag:
                    # Avoid those points which will take the step towards the inflated obstacles
                    obs_threshold_cnt += 1
                    threshold_marker[action] += 1
                    continue
                # Use center of two objects to calculate the geometric distance
                distance = np.sqrt((s_next[0] - gx) ** 2 + (s_next[1] - gy) ** 2)
                if action not in act_dis_dict.keys():
                    act_dis_dict[action] = []
                act_dis_dict[action].append(distance)
            

        min_fail_rate = min(threshold_marker.values()) / run_times
        final_threshold = failure_tolerance + min_fail_rate
        act_dis_dict_all = {action: sum(dis_list) / len(dis_list) for action, dis_list in act_dis_dict.items()}
        for action in act_dis_dict.keys():
            if threshold_marker[action] / run_times >= final_threshold:
                act_dis_dict_all.pop(action)
        if len(act_dis_dict_all) == 0:
            rerun_flag = True
            continue

        best_action = min(act_dis_dict_all, key=act_dis_dict_all.get)
        s_next = move_real(state, best_action, ts=config['sample_rate'], ws_size=workspace, take_noise=True, set_lab_noise=config['use_lab_noise_setup'])
        accept_action_dict[tuple(state)] = (tuple(s_next), best_action)
        action_index = action_map[best_action]
        tmp += [gx, gy, action_index, best_action[0], best_action[1], s_next[0], s_next[1], s_next[2]]
        data.append(tmp)

        if not os.path.exists(file_name):
            pd.DataFrame(data).to_csv(file_name, header=False, index=False)
        else:
            with open(file_name, 'a', newline='') as fd:
                writer = csv.writer(fd)
                writer.writerows(data)
        i += 1
    logging.info("Data Preparation Completed. Time: {} mins".format(int((time.time() - start_time) / 60)))

    action_map[(0, 0)] = act_num
    dict_name = os.path.join(data_folder, "action_map.pkl")
    f = open(dict_name, "wb")
    pickle.dump(action_map, f)
    f.close()

    for k,v  in action_map.items():
        logging.info("({}, {})".format(k, v))

load_env()
