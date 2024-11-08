import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import networkx as nx
import time
import csv
from itertools import product
import logging
from SAC.train_helper import check_folder
import xmltodict
import pickle
from LTL2DRA.dra_utils import *
from SAC.tools import *
import shutil
import math
from config.config import load_config
from utils.tool import *


def load_env():
    start_time = time.time()
    config_name = "./config/ground_robot.yaml"
    config = load_config(config_name)
    data_folder = config['data_folder']
    if os.path.exists(data_folder):
        # remove previous directory
        shutil.rmtree(data_folder)
    check_folder(data_folder)
    log_name = os.path.join(data_folder, "data_collection.log")
    shutil.copy(config_name, data_folder)

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
    shutil.copy(model_sdf, data_folder)
    with open(model_sdf, 'r', encoding='utf-8') as file:
        mysdf = file.read()
    sdf_dict = xmltodict.parse(mysdf)
    sdf_dict = sdf_dict['sdf']['model']['link']
    vis_list = sdf_dict['visual']

    obstacles = []  # goals not included
    goal_size = 0
    inwall_obstacles = []  # obstacles excluded walls
    tmp = []

    """
    extract obstacles from sdf file
    """
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
                inwall_obstacles.append(tmp)
            elif 'box' in vis_list[ind]['geometry'].keys():
                m = list(map(float, vis_list[ind]['pose'].split(" ")))
                box = list(map(float, vis_list[ind]['geometry']['box']['size'].split(" ")))
                tmp = [m[0], m[1], box[0], box[1]]  # [x, y, width, height]
                obstacles.append(tmp)
                inwall_obstacles.append(tmp)
            else:
                continue
        logging.info("{}: {}".format(name, tmp))
    """
    setup multiple goal list 
    (each goal is the center of a grid)
    grid_step: half size of each grid
    """
    workspace_origin = config['ws_size']
    goal_nums = config['goal_nums']
    agent_size = config['agent_size']

    workspace = workspace_origin - config['max_v']  # real inner box workspace
    logging.info("Actual workspace = workspace - max_v * t_s: {}".format(workspace))

    grid_step = workspace / goal_nums  # each grid: 2 * grid_step x 2 * grid_step
    logging.info("grid_step: {}".format(grid_step))
    goal_to_pick_index = max(1, math.ceil(config['max_v'] / (2 * grid_step)))
    logging.info("\nNumber_{} will be picked".format(goal_to_pick_index))
    logging.info("grid_size: {} | max_v: {}".format(2 * grid_step, config['max_v']))

    """
    Pick the goals goal_to_pick_index steps away to make sure the car is going for meaningful goals
    """
    start = -1 * (workspace - grid_step)
    stop = workspace - grid_step
    result = list(np.linspace(start, stop, goal_nums))
    goal_list_pre = list(product(result, result))
    logging.info("\nresult:\n{}\ngrid_step: {}".format(result, grid_step))
    logging.info(result[1] - result[0])
    logging.info(result)

    """
    setup discrete grid world in networkx
    index_graph: based 2d grid world with index denoted as each grid point
    # G[(0, 0)][(0, 1)]['weight'] = 3
    """
    index_graph = nx.grid_2d_graph(goal_nums, goal_nums).to_directed()
    grid_index_map = {}
    assert len(list(index_graph.nodes())) == len(goal_list_pre)
    for i, elem in enumerate(list(index_graph.nodes())):
        grid_index_map[elem] = goal_list_pre[i]
    gridworld = nx.relabel_nodes(index_graph, grid_index_map)
    # plot_weighted_graph(gridworld, name='init.png')

    goal_list = []
    logging.info("Workspace:{} | nums:{}".format(workspace, len(result)))
    for g in goal_list_pre:
        flag = False
        for elm in obstacles:
            if grid_world_inspect(elm, g[0], g[1], grid_step):  # or all_collision_inspect(elm, g[0], g[1]):
                flag = True
                gridworld.remove_nodes_from([g])
                break
        if not flag:
            goal_list.append(g)
    no_edge_node = []
    for node in gridworld.nodes():
        if len(gridworld.edges(node)) == 0:
            """
            If a grid_center is left with no edges, remove it as well
            """
            logging.info("No edges, removing node: {}".format(node))
            no_edge_node.append(node)
    for node in no_edge_node:
        gridworld.remove_nodes_from([node])
        goal_list.remove(node)
    plot_weighted_graph(gridworld, name='basic_grid.png')

    if not nx.is_strongly_connected(gridworld):
        raise Exception("The pruned grid world is not connected, please create a finer grid")
    assert len(list(gridworld.nodes())) == len(goal_list)
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
    logging.info("\n")

    """
    Calculate node weight
    for each grid center
    assign it with 360 cases where each case represent 1 degree of angle
    Then for each case, say if we have 10 actions, apply each action 100 times.
    So in this case we will have 360 * 10 * 100 new nodes 
    we see how many of them ended up crashing into obstacles. 
    That number will be our average 'uncertainty'
    
    update:
    uncertainty map should not actually measure the walls
    """
    angle_sep = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    grid_uncertainty_map = {}
    for grid_center in goal_list:
        grid_uncertainty_map[grid_center] = 0

    run_times = config['uncertain_measure_times']
    total_nums_each_grid = len(dis_action) * run_times * len(angle_sep)

    risky_area_cnt = {}

    t0 = time.time()
    logging.info("Calculating uncertainty map")
    for grid_center in goal_list:
        logging.info("current grid center: {}".format(grid_center))
        crash_counter = 0
        for angle in angle_sep:
            s0 = [grid_center[0], grid_center[1], angle]
            for action in dis_action:
                t = 0
                while t < run_times:
                    s1 = move_real(s0, action, ts=config['sample_rate'], ws_size=workspace_origin,
                                   take_noise=True,
                                   set_lab_noise=config['use_lab_noise_setup'])
                    for elm in obstacles:
                        """
                        If crashing into obstacles
                        s0 (current grid center) will add 1 uncertainty measure value
                        """
                        if all_collision_inspect(elm, s1[0], s1[1], agent_size):
                            crash_counter += 1
                            grid_uncertainty_map[grid_center] += 1
                            break
                    t += 1
        risky_area_cnt[grid_center] = crash_counter
    logging.info("Uncertainty map calculation time: {}".format(time.time() - t0))
    logging.info("\n\nTotal: {}".format(total_nums_each_grid))
    for k, v in risky_area_cnt.items():
        logging.info("{}---> {}".format(k, v))

    for key in grid_uncertainty_map.keys():
        grid_uncertainty_map[key] /= total_nums_each_grid
    logging.info("\nUncertainty map:")
    for k, v in grid_uncertainty_map.items():
        logging.info("{}: {}".format(k, v))

    """
    add edge weight to the graph
    """
    for v_start in gridworld.nodes():
        x_start, y_start = v_start
        for v_end in gridworld.nodes():
            if gridworld.has_edge(v_start, v_end):
                x_end, y_end = v_end
                l2_dis = np.sqrt((x_start - x_end) ** 2 + (y_start - y_end) ** 2)
                if grid_uncertainty_map[v_end] >= 0.5:
                    # very unsafe area
                    gridworld[v_start][v_end]['weight'] = 10
                else:
                    gridworld[v_start][v_end]['weight'] = grid_uncertainty_map[v_end] * l2_dis

    # ###############################################################################
    # Test code
    ###################
    # for edge in gridworld.edges.data('weight'):
    #     print(edge, "\n")

    # state_list = [
    #     # [-1.27, 0, 0],
    #     # [-1.27, 0, 1.57],
    #     # [-1.27, 0, 3.14],
    #     # [-1.27, 0, -1.57],
    #     # [0, 1.2, 0],
    #     # [0, 1.2, 1.57],
    #     # [0, 1.2, 3.14],
    #     [0, 1.1573333333333333, 0]
    # ]
    # goals = [
    #     (-1.1458333333333333, -0.5208333333333333),
    # ]
    # print("\n--------------------------\n")
    # for goal in goals:
    #     for state in state_list:
    #         print("\nstate: {}".format(state))
    #         near_dis = float('inf')
    #         nearest_grid = None
    #         for grid in goal_list:
    #             l2 = np.sqrt((state[0] - grid[0]) ** 2 + (state[1] - grid[1]) ** 2)
    #             if l2 <= near_dis:
    #                 nearest_grid = grid
    #                 near_dis = l2
    #         print("nearest grid: {}\ngoal:{}".format(nearest_grid, goal))
    #         if all_collision_inspect(list(goal)+[goal_size], state[0], state[1], agent_size):
    #             print("remain unchanged")
    #             continue
    #         length, path = nx.bidirectional_dijkstra(gridworld, nearest_grid, (goal[0], goal[1]))
    #         # print('\ncurrent length: {} | goal:{} | path: {}'.format(length, (goal[0], goal[1]), path))
    #         print(path)
    #         """
    #         TIME difference in ROSPY? how to solve the reaction time difference
    #         """

    #         index = min(len(path)-1, goal_to_pick_index)
    #         print("the place you should go next: {}".format(path[index]))
    #         real_cur_goal = path[index]
    #         gx, gy = real_cur_goal
    #         accept_action_dict = {}
    #         act_dis_dict = {}
    #         run_times = config['data_run_times']
    #         tmp = state
    #         # if a certain percentage of points has collied with obstacles, reject them
    #         failure_tolerance = config['bias_fail_tolerance']
    #         threshold_marker = {action: 0 for action in dis_action}
    #         for action in dis_action:
    #             t = 0
    #             obs_threshold_cnt = 0
    #             while t < run_times:
    #                 # Do not break the loop to avoid all empty set
    #                 t += 1
    #                 s_next = move_real(state, action,
    #                                 ts=config['sample_rate'],
    #                                 ws_size=workspace_origin,
    #                                 take_noise=True,
    #                                 set_lab_noise=config['use_lab_noise_setup'])
    #                 # print(s_next)
    #                 flag = False
    #                 for elm in obstacles:
    #                     if all_collision_inspect(elm, s_next[0], s_next[1], agent_size):
    #                         flag = True
    #                         break
    #                 if flag:
    #                     # Avoid those points which will take the step towards the inflated obstacles
    #                     obs_threshold_cnt += 1
    #                     threshold_marker[action] += 1
    #                     continue
    #                 # Use center of two objects to calculate the geometric distance
    #                 distance = np.sqrt((s_next[0] - gx) ** 2 + (s_next[1] - gy) ** 2)
    #                 if action not in act_dis_dict.keys():
    #                     act_dis_dict[action] = []
    #                 act_dis_dict[action].append(distance)

    #         for k,v in threshold_marker.items():
    #             print(k, v)
    #         min_fail_rate = min(threshold_marker.values()) / run_times
    #         final_threshold = failure_tolerance + min_fail_rate
    #         act_dis_dict_all = {action: sum(dis_list) / len(dis_list) for action, dis_list in act_dis_dict.items()}
    #         for action in act_dis_dict.keys():
    #             if threshold_marker[action] / run_times >= final_threshold:
    #                 act_dis_dict_all.pop(action)
    #         if len(act_dis_dict_all) == 0:
    #             continue
    #         best_action = min(act_dis_dict_all, key=act_dis_dict_all.get)
    #         print("action to choose: {}".format(best_action))
    #         s_next = move_real(state,
    #                         best_action,
    #                         ts=config['sample_rate'],
    #                         ws_size=workspace_origin,
    #                         take_noise=True,
    #                         set_lab_noise=config['use_lab_noise_setup'])
    #         print("after 1s: {}\n-------------------\n".format(s_next))
    # exit(0)
    ###############################################################################################################

    goal_change = config['data_collect_goal_change']
    i = 0
    ind = 0
    rerun_flag = False
    cur_goal = None
    action_index = 0

    while True:
        # change goal every N rounds
        if i % goal_change == 0 and not rerun_flag:
            if ind >= len(goal_list):
                print("Data collection completed, exiting.")
                break
            cur_goal = goal_list[ind]
            ind += 1
            logging.info(
                "i:{} | ind:{} | Current goal is: {}| Time: {}".format(i, ind, cur_goal, time.time() - start_time))
        if rerun_flag:
            rerun_flag = False

        state = reset_obs_free_risk_aware(obstacles, workspace, agent_size)

        """
        Check if goals and states hits in the first place
        if yes, then just remain in the goal without moving
        """
        cur_goal_with_radius = list(cur_goal)
        cur_goal_with_radius.append(goal_size)
        tmp_goal = state
        data_goal = []
        if all_collision_inspect(cur_goal_with_radius, state[0], state[1]):
            tmp_goal += [cur_goal[0], cur_goal[1], action_index, 0, 0, state[0], state[1], state[2]]
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
        assign the state to the nearest grid center to calculate shortest path within grid world
        """
        # print("state: {}".format(state))
        near_dis = float('inf')
        nearest_grid = None
        for grid in goal_list:
            l2 = np.sqrt((state[0] - grid[0]) ** 2 + (state[1] - grid[1]) ** 2)
            if l2 < near_dis:
                nearest_grid = grid
                near_dis = l2
        # print("nearest grid: {}".format(nearest_grid))

        """
        Find shortest path for current state_grid to the goal
        """
        length, path = nx.bidirectional_dijkstra(gridworld, nearest_grid, cur_goal)
        # print('current length: {} | goal:{} | path: {}'.format(length,cur_goal, path))
        # print("the place you should go next: {}".format(path[1]))

        if len(path) == 1:
            """
            current state is classified into the same grid as the current goal
            """
            real_cur_goal = path[0]
        elif len(path) > 1:
            """
            go to the next state that will take you to goal while keeping you safe
            """
            index = min(len(path) - 1, goal_to_pick_index)
            real_cur_goal = path[index]
        else:
            """ the length of the shortest path is equal to 0"""
            raise Exception("The graph might not be connected. Please check your graph")

        """
        change to the nearest safe goal if we did not start from the actual goal
        """
        gx_real, gy_real = real_cur_goal

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
                s_next = move_real(state, action, ts=config['sample_rate'], ws_size=workspace_origin, take_noise=True,
                                   set_lab_noise=config['use_lab_noise_setup'])
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
                distance = np.sqrt((s_next[0] - gx_real) ** 2 + (s_next[1] - gy_real) ** 2)
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
        s_next = move_real(state, best_action, ts=config['sample_rate'], ws_size=workspace_origin, take_noise=True,
                           set_lab_noise=config['use_lab_noise_setup'])

        # logging.info("\nG:({:.2f}, {:.2f}) | grid:({:.2f}, {:.2f}) | Our:({:.2f}, {:.2f}) | ({:.2f}, {:.2f}): a[{:.2f}, {:.2f}] -> ({:.2f}, {:.2f})".format(
        #     cur_goal[0], cur_goal[1],
        #     nearest_grid[0], nearest_grid[1],
        #     real_cur_goal[0], real_cur_goal[1],
        #     state[0], state[1], 
        #     best_action[0], best_action[1], 
        #     s_next[0], s_next[1]))

        accept_action_dict[tuple(state)] = (tuple(s_next), best_action)
        action_index = action_map[best_action]
        tmp += [cur_goal[0], cur_goal[1], action_index, best_action[0], best_action[1], s_next[0], s_next[1], s_next[2]]
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

    for k, v in action_map.items():
        logging.info("({}, {})".format(k, v))


load_env()
