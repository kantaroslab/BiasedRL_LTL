import numpy as np
import os
import pandas as pd
import random
from helicopter_gym.env import ChopperScape
from shapely.geometry import Polygon
import time
import csv
from itertools import product
import logging
from SAC.train_helper import check_folder
import visilibity as vis


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


def move(xv, uv, ts, env, take_noise=True):
    # convert from pixels/sec to m/sec
    xv = [xv[0] / 100, xv[1] / 100, xv[2]]
    uv = [uv[0] / 100, uv[1] / 100]
    state_next = dynamics(xv, uv, ts)
    if take_noise:
        process_noise = (
                0.01 * np.dstack(
            [-np.ones(3), np.ones(3)]
        )[0]
        )
        xv_ = np.array(xv)
        noise = np.random.uniform(
            low=process_noise[:, 0],
            high=process_noise[:, 1],
            size=xv_.shape,
        )
        state_next += noise
    state_next = [state_next[0] * 100, state_next[1] * 100, state_next[2]]
    xp = clamp(state_next[0], env.x_min, env.x_max)
    yp = clamp(state_next[1], env.y_min, env.y_max)
    tp = clamp(state_next[2], 0, 2 * np.pi)
    return np.array([xp, yp, tp])


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


def setup_vis_env(env):
    epsilon = 0.0001

    maxx, maxy = env.observation_shape[0], env.observation_shape[1]
    ox, oy, _ = env.bird.get_position()  # obstacle
    ox2, oy2, _ = env.bird2.get_position()  # another obstacle
    ox_w, ox_h = env.bird.icon_w, env.bird.icon_h

    # Define outer boundary
    p1 = vis.Point(0, 0)
    p2 = vis.Point(maxx, 0)
    p3 = vis.Point(maxx, maxy)
    p4 = vis.Point(0, maxy)
    walls = vis.Polygon([p1, p2, p3, p4])

    # Define obstacle 1
    p1 = vis.Point(ox, oy)
    p2 = vis.Point(ox, oy + ox_h)
    p3 = vis.Point(ox + ox_w, oy + ox_h)
    p4 = vis.Point(ox + ox_w, oy)
    obstacle1 = vis.Polygon([p1, p2, p3, p4])

    # Define obstacle 2
    p1 = vis.Point(ox2, oy2)
    p2 = vis.Point(ox2, oy2 + ox_h)
    p3 = vis.Point(ox2 + ox_w, oy2 + ox_h)
    p4 = vis.Point(ox2 + ox_w, oy2)
    obstacle2 = vis.Polygon([p1, p2, p3, p4])

    res_env = vis.Environment([walls, obstacle1, obstacle2])
    if walls.is_in_standard_form() and \
            obstacle1.is_in_standard_form() and \
            obstacle2.is_in_standard_form() and \
            res_env.is_valid(epsilon):
        print("VisiLibity Setup Successfully")
    return res_env, epsilon


def load_env():
    os.chdir(os.path.dirname(__file__))
    DATA_CODE = 1  # decide which NN data to collect.

    data_folder = './dataset'
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

    seed = 0
    np.random.seed(seed)
    env = ChopperScape()
    env.reset()

    """
    Choose random action from finite discrete action space
    For each action, apply to current action for 500 times
    calculate the average distance towards final state
    Iterate for all actions, choose the biased one that brings us closest to the goal
    No planning algorithm is needed (?)

    By Yiannis: Given x0, xf, apply MDP(x0,a) multiple times (e.g, 500 times), for each action, to see which actions 
    brings you more often closer to xf. That’s the biased action a_b (i.e., dist(x_next,xf)<dist(x0,xf) when a_b is 
    applied at x0) that you should use for training 
    """

    # get environment basic info
    ox, oy, _ = env.bird.get_position()  # obstacle
    ox2, oy2, _ = env.bird2.get_position()  # another obstacle
    gx, gy, _ = env.goal.get_position()  # goal
    ox_w, ox_h = env.bird.icon_w, env.bird.icon_h
    gx_w, gx_h = env.goal.icon_w, env.goal.icon_h
    assert ox_h == ox_w
    assert gx_w == gx_h

    # Update: Setup visilibity
    vis_env, vis_eps = setup_vis_env(env)

    # setup multiple goal list
    goal_part_step = 30
    max_x = env.observation_shape[0]
    max_y = env.observation_shape[1]
    start = 0
    step = goal_part_step
    stop = max_x - env.goal.icon_w
    result_x = np.linspace(start, stop, int((stop - start) / step + 1))
    stop = max_y - env.goal.icon_h
    result_y = np.linspace(start, stop, int((stop - start) / step + 1))
    goal_list_pre = list(product(result_x, result_y))
    goal_list = []
    assert env.goal.icon_w == env.goal.icon_h
    goal_size = env.goal.icon_w
    for elm in goal_list_pre:
        # remove those goals which has intersection with the obstacles
        if not collision_inspect(elm[0], elm[1], goal_size, ox, oy, ox_w) and \
                not collision_inspect(elm[0], elm[1], goal_size, ox2, oy2, ox_w):
            goal_list.append(elm)
    logging.info("Available goals: {}".format(len(goal_list)))

    # discretize the action space
    dis_v = list(np.linspace(-22, 22, 6))
    dis_w = list(np.linspace(-15, 15, 5))
    dis_action = list(product(dis_v, dis_w))
    action_map = {}
    logging.info("\nNumber of discrete actions: {}\n".format(len(dis_action)))
    # Update: add zero action
    # Update2: do not add zero action into consideration for data collection unless it starts from goal
    # dis_action.append((0.0, 0.0)) this action should be put after the whole procedure is done

    num = 0
    for act in dis_action:
        logging.info(act)
        action_map[act] = num
        num += 1

    import pickle
    dict_name = os.path.join(data_folder, "action_map.pkl")
    f = open(dict_name, "wb")
    pickle.dump(action_map, f)
    f.close()

    # Update: inflate the obstacle
    ox_, oy_, ox_w_ = inflate_obstacle(ox, oy, ox_w, env)
    ox2_, oy2_, ox_w_ = inflate_obstacle(ox2, oy2, ox_w, env)

    goal_change = 1000
    i = 0
    ind = 0
    start_time = time.time()
    pause = False
    while True:
        # change goal every N rounds
        if i % goal_change == 0 and not pause:
            if ind >= len(goal_list):
                print("Data collection completed, exiting.")
                break
            cur_goal = goal_list[ind]
            ind += 1
            c = goal_size / 2
            gx, gy = cur_goal[0] + c, cur_goal[1] + c  # use the center of goal as comparison
            logging.info(
                "i:{} | Current goal selection is: ({}, {}) | Time: {}".format(i, gx, gy, time.time() - start_time))

        # setup initial state
        init_x = random.uniform(0, env.observation_shape[0])
        init_y = random.uniform(0, env.observation_shape[1])
        init_theta = random.uniform(0, np.pi * 2)
        state = [init_x, init_y, init_theta]
        agent_size = env.agent_icon
        # Update: initial system state is not allowed to be inside obstacle regions
        # But it is ok if we initial the system inside the goal region
        if collision_inspect(init_x, init_y, agent_size, ox_, oy_, ox_w_) or \
                collision_inspect(init_x, init_y, agent_size, ox2_, oy2_, ox_w_):
            if i % goal_change == 0:
                pause = True
            else:
                pause = False
            continue
        pause = False

        # if the state chosen is feasible
        """
        data structure: 
        [x0, y0, t0, xf, yf, index, action[0], action[1], x1, y1, t1]
        """
        data = []
        accept_action_dict = {}
        """
        Major update objective - 2 different biased selection network policy
        1: biased towards directions that minimize distance to goal
        For each action:
            For number of run_times, calculate the average distance to the goal
            (IMPORTANT): if a certain percentage of points has collied with obstacles, reject them
            Select the one with the smallest average distance
        
        2: biased towards safe directions
        For each action:
            For number of run_times, first setup the overall threshold of 'being safe'
            Calculate how many times each action is being safe during run_times
            Collect all the actions which have satisfied the 'safe threshold'
            Select the action from the collected action set with the smallest distance to the goal    
            
        Probably this can be mixed together
        
        --- UPDATE ON NOV 26, 2022 ---
        Set up minimum threshold
        Do not throw away those 'unsafe' actions
        Record all the lowest threshold
        Real threshold = min + delta
        """

        # Update: removed 0 action and keep applying normal policy to move agent
        act_dis_dict = {}
        run_times = 1
        tmp = state
        # reject_threshold = 0.1
        # if a certain percentage of points has collied with obstacles, reject them
        failure_tolerance = 0.2
        threshold_marker = {action: 0 for action in dis_action}
        for action in dis_action:
            t = 0
            obs_threshold_cnt = 0
            while t < run_times:
                # Do not break the loop to avoid all empty set
                t += 1
                s_next = move(xv=state, uv=action, ts=1, env=env)
                if collision_inspect(s_next[0], s_next[1], agent_size, ox_, oy_, ox_w_) or \
                        collision_inspect(s_next[0], s_next[1], agent_size, ox2_, oy2_, ox_w_):
                    # Avoid those points which will take the step towards the inflated obstacles
                    obs_threshold_cnt += 1
                    threshold_marker[action] += 1
                    continue
                # Use center of two objects to calculate the geometric distance
                vis_cur_state = [s_next[0] + agent_size / 2, s_next[1] + agent_size / 2]
                vis_goal_state = [gx + gx_w / 2, gy + gx_h / 2]
                # distance = np.sqrt((s_next[0] - gx) ** 2 + (s_next[1] - gy) ** 2)
                distance = check_vis_distance(vis_cur_state, vis_goal_state, vis_env, vis_eps)
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
            print("State: {}".format(state))
            print("Obstacles: ", ox_, " ", oy_, " ", ox2_, " ", oy2_)
            print(threshold_marker)
            for action in dis_action:
                s_next = move(xv=state, uv=action, ts=1, env=env)
                if collision_inspect(s_next[0], s_next[1], agent_size, ox_, oy_, ox_w_) or \
                        collision_inspect(s_next[0], s_next[1], agent_size, ox2_, oy2_, ox_w_):
                    print("s_next: {} | Bumped".format(s_next))
            continue

        best_action = min(act_dis_dict_all, key=act_dis_dict_all.get)
        s_next = move(xv=state, uv=best_action, ts=1, env=env, take_noise=True)
        while collision_inspect(s_next[0], s_next[1], agent_size, ox_, oy_, ox_w_) or \
                collision_inspect(s_next[0], s_next[1], agent_size, ox2_, oy2_, ox_w_):
            # (Just in case) get a state pair which does not hit obstacle
            s_next = move(xv=state, uv=best_action, ts=1, env=env, take_noise=True)
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


def check_vis_distance(state, goal, vis_env, epsilon):
    observer = vis.Point(state[0], state[1])
    end = vis.Point(goal[0], goal[1])
    observer.snap_to_boundary_of(vis_env, epsilon)
    observer.snap_to_vertices_of(vis_env, epsilon)
    shortest_path = vis_env.shortest_path(observer, end, epsilon)
    route = shortest_path.path()
    distance = 0
    prev_x, prev_y = route[0].x(), route[1].y()
    for ind in range(1, len(route)):
        cur_x, cur_y = route[ind].x(), route[ind].y()
        distance += np.sqrt((cur_x - prev_x) ** 2 + (cur_y - prev_y) ** 2)
        prev_x, prev_y = cur_x, cur_y
    return distance


def inflate_obstacle(x, y, size, env):
    x_max = env.observation_shape[0]
    y_max = env.observation_shape[1]
    inflate_size = int(size / 5)
    x_new = max(min(x - inflate_size, x_max), 0)
    y_new = max(min(y - inflate_size, y_max), 0)
    new_size = size + inflate_size
    return x_new, y_new, new_size


load_env()
