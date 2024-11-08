import torch
import os
import logging
import numpy as np
import pickle
import random
from itertools import product
from SAC.sac_net import *
import torch.nn as nn
from utils.tool import ReplayBuffer as DQNReplayBuffer
from utils.network import DQN
from SAC.train_helper import Network, normalize_data_ros, check_folder
from LTL2DRA.translate_ros import rabin_pre_setup_ros, interpret_ros
from LTL2DRA.dra_utils import *
from SAC.tools import all_collision_inspect
import rospy
from gazebo_msgs.srv import *
from datetime import datetime
import xmltodict
import time
global sub_once
import shutil
import glob
import pandas as pd
import matplotlib.pyplot as plt
import torch 
from config.config import load_config
from torch.utils.tensorboard import SummaryWriter
import subprocess
import csv

config_name = "./config/ground_robot.yaml"
config = load_config(config_name)


def initial_configuration(folder_name, output_folder):
    logging.info("\n[Biased Network Setup]")
    action_map_name = os.path.join(folder_name, "action_map.pkl")
    with open(action_map_name, "rb") as f:
        data = pickle.load(f)
    action_map = {}
    for k, v in data.items():
        action_map[v] = k
    INPUT_SIZE = config['bias_input']
    logging.info("Number of discrete actions: {}".format(len(action_map)))
    OUTPUT_SIZE = len(action_map)
    net_dims = [INPUT_SIZE] + config['bias_net_dim'] + [OUTPUT_SIZE]

    model_name = os.path.join(folder_name, config['selected_bias_model'])
    logging.info("Biased Network loaded from {}".format(model_name))
    model = Network(net_dims, activation=nn.ReLU).net
    logging.info("Biased Network Structure: {}".format(net_dims))
    model.load_state_dict(torch.load(model_name))
    model.eval()
    logging.info("Biased Network Setup Completed")

    logging.info("Copying related files for backup purpose.")
    shutil.copy(model_name, output_folder)
    shutil.copy(action_map_name, output_folder)
    origin_data = os.path.join(folder_name, "data.csv")
    shutil.copy(origin_data, output_folder)
    return model, action_map


def Evaluate_in_Gazebo(model, max_eval_steps, action_map, agent_dqn, rabin, obstacles, workspace, goal_list, goal_size, ap_range_map):
    episode_steps = 0
    accept_counter = 0
    done = False
    rate = rospy.Rate(1)
    s = reset_obs_free(obstacles, workspace)
    reset_gazebo_with_designed_state(s)
    rate.sleep()

    cur_mix = list(s) + [rabin.init_state]
    logging.info("Gazebo turtlebot angle ranges from [-pi, pi], hence we modify the range [0, 2pi] to [-pi, pi]")

    (gx, gy, gsize), goal_ap = rabin.select_next_rabin(cur_mix, rabin, goal_list, goal_size, ap_range_map)
    
    eva_reward = 0
    while episode_steps <= max_eval_steps:
        action_ind = agent_dqn.take_action_from_net(cur_mix)
        a = list(action_map[action_ind]) 

        # print("Goal ap: {}".format(goal_ap))
        # cur_input_ = np.array([s[0], s[1], s[2], gx, gy])
        # cur_input = torch.FloatTensor(normalize_data_ros(cur_input_, workspace)).unsqueeze(0)
        # output = model(cur_input)
        # _, pred = torch.max(output, 1)
        # pred_ind = pred.detach().numpy()[0]
        # a = list(action_map[pred_ind])

        a = [a[0], a[1]]  # Gazebo works from -pi ~ pi
        publish_action_to_gazebo(a, is_noisy=True)
        rate.sleep()
        s_ = get_cur_gazebo_model_state()
        next_mix, r, done, accept_counter = interpret_ros(cur_mix, s_, rabin, accept_counter)
        eva_reward += r
        if next_mix[-1] != cur_mix[-1] and next_mix[-1] not in rabin.deadlock:
            (gx, gy, _), goal_ap = rabin.select_next_rabin(next_mix, rabin, goal_list, goal_size,
                                                                        ap_range_map)
            if goal_ap != "Any":
                    logging.info("Changing goal to: {}".format(ap_range_map[goal_ap]))
            else:
                logging.info("Current (nearest) goal center: ({}, {})".format(gx, gy))
        if episode_steps == max_eval_steps and done is not True:
                done = True
        print("M_State:({:.4f}, {:.4f}, {:.4f}, {}) | r:{} | OK:{}".format(next_mix[0],next_mix[1],next_mix[2],next_mix[3], r, accept_counter))
        cur_mix = next_mix
        s = s_
        episode_steps += 1
        if done:
            break
    return eva_reward

def clamp(n, smallest, largest):
    for i in range(len(n)):
        n[i] = max(smallest[i], min(n[i], largest[i]))
    return n

def main(output_folder, writer):
    logging.info("------------\nDQN + BIASED_NET + ROS + GAZEBO\n------------")


    # New Setup: Use a single AP to define all the obstacles 
    logging.info("\n[LTL Specification]")
    LTL_formula = config['ltl_task']
    logging.info("LTL Formula={}".format(LTL_formula))

    shutil.copy(config_name, output_folder)

    backup_folder = os.path.join(output_folder, 'files')
    check_folder(backup_folder)
    model, action_map = initial_configuration(config['selected_bias_folder'], backup_folder)

    max_episode_steps = config['max_episode_steps']
    max_eval_steps = config['max_eval_steps']  # evaluate within gazebo
    state_dim = config['robot_dim']
    action_dim = config['action_dim']
    max_action = np.array([config['max_v'], config['max_w']])
    logging.info("\n[Environment Spec Setup]")
    logging.info("state_dim={}".format(state_dim))
    logging.info("action_dim={}".format(action_dim))
    logging.info("max_action={}".format(max_action))
    logging.info("max_episode_steps={}".format(max_episode_steps))

    logging.info("\n[RL Agent Setup]")
    mix_dim = state_dim + 1
    dqn_replay_buffer = DQNReplayBuffer(capacity=config['dqn_buffer_size'])
    dqn_action_dim = len(action_map)
    # Setup DQN
    agent_dqn = DQN(mix_dim, config['dqn_hidden'], dqn_action_dim, learning_rate=config['rl_lr'],
                    gamma=config['discount_factor'], target_update=config['target_update'], 
                    reward_clip=config['reward_clip'], use_soft_update=config['use_soft_update'], 
                    tau=config['dqn_tau'])
    logging.info("Mix dimension={} -> [x, y, theta, dra_state]".format(mix_dim))
    logging.info("Learning rate={}".format(config['rl_lr']))
    logging.info("DQN Network Structure: {}".format(agent_dqn.q_net))

    logging.info("\n[Training Spec Setup]")
    evaluate_freq = config['rl_eval_freq']  # Evaluate the policy every 'evaluate_epi' episode
    total_steps = 0  # Record the total steps during the training
    max_episodes = config['rl_max_episodes']
    epsilon = float(config['epsilon'][0])
    epsilon_min = float(config['epsilon'][1])
    epsilon_decay_steps = config['epsilon'][2]
    epsilon_decay = (epsilon - epsilon_min) / epsilon_decay_steps
    delta = float(config['delta'][0]) 
    # delta_min = float(config['delta'][1]) 
    # delta_decay_steps = config['delta'][2]
    # delta_decay = (delta - delta_min) / delta_decay_steps
    delta_max = float(config['delta'][1]) 
    delta_grow_steps = config['delta'][2]
    delta_grow = (delta_max - delta) / delta_grow_steps
    
    logging.info("Epsilon={} | Epsilon_Min={} | Decay={}".format(epsilon, epsilon_min, epsilon_decay))
    # logging.info("Delta={} | Delta_Min={} | Decay={}".format(delta, delta_min, delta_decay))
    logging.info("Delta={} | Delta_Max={} | Grow={}".format(delta, delta_max, delta_grow))
    logging.info("Max episode={}".format(max_episodes))

    logging.info("\n[Gazebo World Config]")
    model_sdf = config['world_model_sdf']
    shutil.copy(model_sdf, backup_folder)
    logging.info("Loading SDF from: {}".format(model_sdf))
    with open(model_sdf, 'r', encoding='utf-8') as file:
        mysdf = file.read()
    sdf_dict = xmltodict.parse(mysdf)
    sdf_dict = sdf_dict['sdf']['model']['link']
    vis_list = sdf_dict['visual']

    logging.info("Name of objects in Gazebo World:")
    for ind in range(len(vis_list)):
        name = vis_list[ind]['@name']
        logging.info(name)
    obs_list = config['obstacle_list']
    logging.info("obstacles are: {}".format(obs_list))
    # for key, value in sdf_dict.items():
    #     logging.info("{}, {}\n".format(key, value))

    obstacles = []  # goals not included
    ap_range_map = {}  # goals included
    goal_size = 0
    tmp = []
    for ind in range(len(vis_list)):
        # Check the obs_list carefully, make sure all obstacles are included.
        # For cylinder-obstacle or goal: [x, y, radius]
        # For walls: [x_center, y_center, w, h]
        name = vis_list[ind]['@name']
        if name.startswith("goal"):
            # goals do not have collision volume
            # objects with collision volume cannot be "reached" by robot
            goal_size = float(vis_list[ind]['geometry']['cylinder']['radius'])
            m = list(map(float, vis_list[ind]['pose'].split(" ")))
            radius = float(vis_list[ind]['geometry']['cylinder']['radius'])
            tmp = [m[0], m[1], radius]
            ap_range_map[name] = tmp
        elif name.startswith("wall"):
            m = list(map(float, vis_list[ind]['pose'].split(" ")))
            box = list(map(float, vis_list[ind]['geometry']['box']['size'].split(" ")))
            tmp = [m[0], m[1], box[0], box[1]]  # [x, y, width, height]
        elif name.startswith("obs"):
            if 'cylinder' in vis_list[ind]['geometry'].keys():
                m = list(map(float, vis_list[ind]['pose'].split(" ")))
                radius = float(vis_list[ind]['geometry']['cylinder']['radius'])
                tmp = [m[0], m[1], radius]
            elif 'box' in vis_list[ind]['geometry'].keys():
                m = list(map(float, vis_list[ind]['pose'].split(" ")))
                box = list(map(float, vis_list[ind]['geometry']['box']['size'].split(" ")))
                tmp = [m[0], m[1], box[0], box[1]]  # [x, y, width, height]
            else:
                continue
        if name in obs_list:
            obstacles.append(tmp)
        logging.info("{}: {}".format(name, tmp))
    ap_range_map['obstacles'] = obstacles  # consider all obstacles as 1 AP

    logging.info("\n[Initialize Rabin Automaton]")
    rabin = rabin_pre_setup_ros(ap_range_map, LTL_formula)

    for k, v in ap_range_map.items():
        print(k, v, "\n")

    # move all 'command*' file to output folder
    cmd_files = glob.glob('command*')
    for file in cmd_files:
        shutil.move(file, backup_folder)
    py_files = glob.glob("*.py")
    for file in py_files:
        shutil.copy(file, backup_folder)

    logging.info("\n[Setup discrete world for random selection]")
    workspace_origin = config['ws_size']
    goal_nums = config['goal_nums']
    # goal_part_step = config['goal_part_step']
    agent_size = config['agent_size']  # radius of turtlebot

    workspace = workspace_origin - agent_size
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
            if all_collision_inspect(elm, g[0], g[1], agent_size):
                flag = True
                break 
        if not flag:
            goal_list.append(g)

    logging.info("\n[Training Process Starts]")
    if config['run_eval_in_rl']:
        # No need to start roscore & gazebo if no evaluation during training
        logging.info("\n[Initialize ROS Node for Evaluation Purpose]")
        rospy.init_node('RL_Agent', anonymous=False, log_level=rospy.FATAL)
        logging.getLogger().addHandler(logging.FileHandler(log_name))
    
    cur_epi = 0
    init_time = time.time()
    reward_list = []
    reward_x_axis = []
    dqn_loss_x_axis = []
    dqn_loss_list = []
    train_r_list = []
    epi_r_x_list = []
    eva_counter = 0

    # starts from 0
    writer.add_scalar("Train_Episode_Discount_Reward/train", torch.tensor(0), cur_epi)
    writer.add_scalar("Train_Episode_Discount_Reward/time", torch.tensor(0), cur_epi)


    reward_file_name = os.path.join(output_folder, "discount_episode_reward.csv")
    total_allowed_runtime = config['RL_RUNTIME']

    # while cur_epi <= max_episodes and (time.time() - init_time)/60 <= total_allowed_runtime:
    while cur_epi <= max_episodes:
        s = reset_obs_free_risk_aware(obstacles, workspace_origin, agent_size)
        # Now initialize everywhere in workspace

        episode_steps = 1
        done = False
        cur_epi += 1
        cur_mix = list(s) + [rabin.init_state]

        (gx, gy, gsize), goal_ap = rabin.select_next_rabin(cur_mix, rabin, goal_list, goal_size, ap_range_map)

        # The usage of the accept_counter is to end the episode after
        # the accepting pair has been reached for this number of times
        accept_counter = 0
        cur_dqn_action_index = 0
        dqn_loss = 0

        discount_episode_reward = 0

        reward_data = []
        act_data = []
        while episode_steps <= max_episode_steps:

            # V(s_0) where s_0 is the initial state and V(s_0) = max_{action a} Q(s_0,a)
            value_func = agent_dqn.get_value_function(cur_mix)
            writer.add_scalar("Value_Function/Train", value_func, total_steps)

            ran = np.random.uniform()
            if ran <= 1 - epsilon:
                # select from dqn
                action_ind = agent_dqn.take_action_from_net(cur_mix)
                cur_dqn_action_index = action_ind
                a = list(action_map[action_ind]) 
            else:
                # Apply Biased Network - randomly select from discrete actions
                if ran <= delta:
                    cur_input_ = np.array([s[0], s[1], s[2], gx, gy])
                    cur_input = torch.FloatTensor(normalize_data_ros(cur_input_, workspace_origin)).unsqueeze(0)
                    output = model(cur_input).cuda()
                    _, pred = torch.max(output, 1)
                    pred_ind = pred.detach().cpu().numpy()[0]
                    a = list(action_map[pred_ind])
                else:
                    # random explore
                    choice = random.randint(0, len(action_map)-1)
                    a = [action_map[choice][0], action_map[choice][1]]
            a = clamp(a, -max_action, max_action)
            act_data.append(a)
            s_ = move_real(s, a, ts=config['sample_rate'], ws_size=workspace_origin, take_noise=True, set_lab_noise=config['use_lab_noise_setup'])
            
            # add actual action?
            next_mix, r, done, accept_counter = interpret_ros(cur_mix, s_, rabin, accept_counter)
            discount_episode_reward += (config['discount_factor'] ** episode_steps) * r

            if next_mix[-1] != cur_mix[-1] and next_mix[-1] not in rabin.deadlock:
                # logging.info("Rabin state satisfied from {} to {}".format(cur_mix[-1], next_mix[-1]))
                # rabin state has been changed because one of the AP is satisfied by the system state
                (gx, gy, gsize), goal_ap = rabin.select_next_rabin(next_mix, rabin, goal_list, goal_size, ap_range_map)
                
            if episode_steps == max_episode_steps and done is not True:
                done = True
            dqn_replay_buffer.add(cur_mix, cur_dqn_action_index, r, next_mix, done)
            cur_mix = next_mix
            s = s_

            if cur_epi >= config['dqn_train_signal']:
                b_s, b_a, b_r, b_ns, b_d = dqn_replay_buffer.sample(batch_size=config['dqn_batch'])
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': b_d
                }
                dqn_loss = agent_dqn.update(transition_dict)
                writer.add_scalar("DQN_Loss/train", dqn_loss, total_steps)
                
                dqn_loss = dqn_loss.detach().cpu().numpy()
                dqn_loss_list.append(dqn_loss)
                dqn_loss_x_axis.append(total_steps)

            total_steps += 1
            episode_steps += 1
            if done:
                break

        epi_time = time.time() - init_time


        # write episode reward to csv
        reward_tmp = [cur_epi, discount_episode_reward]
        reward_data.append(reward_tmp)
        with open(reward_file_name, 'a', newline='') as fd:
            csvwriter = csv.writer(fd)
            csvwriter.writerows(reward_data)
        
        writer.add_scalar("Train_Episode_Discount_Reward/train", torch.tensor(discount_episode_reward), cur_epi)
        writer.add_scalar("Train_Episode_Discount_Reward/time", torch.tensor(discount_episode_reward), epi_time)

        delta = delta + delta_grow if delta + delta_grow < delta_max else delta_max
        eps_new = epsilon - epsilon_decay
        if 1-eps_new >= delta:
            epsilon = 1-delta 
        else:
            if epsilon_min < eps_new:
                epsilon = eps_new 
            else:
                epsilon = epsilon_min

        epi_r_x_list.append(cur_epi)
        train_r_list.append(discount_episode_reward)

        logging.info("Episode:{} | R:{:.1f} | Epi_Loss:{:.2f} | 1-ε:{:.2f} | δ:{:.2f} | Time:{:.1f} min".format(cur_epi, discount_episode_reward, dqn_loss, 1-epsilon, delta, epi_time/60))
        
        if cur_epi % evaluate_freq == 0:
            if config['run_eval_in_rl']:
                eva_counter += 1
                logging.info("Initialize evaluation procedure in Gazebo after {} episodes with {} steps".format(cur_epi, total_steps))
                eva_reward = Evaluate_in_Gazebo(model, max_eval_steps, action_map, agent_dqn, rabin, obstacles, workspace_origin, goal_list, goal_size, ap_range_map)
                writer.add_scalar("Episode_Reward/test", torch.tensor(eva_reward), eva_counter)
                reward_x_axis.append(cur_epi)
                reward_list.append(eva_reward)

            # Save PTH model
            logging.info("Saving model at episode: {}".format(cur_epi))
            model_path = os.path.join(output_folder, "model_checkpoints")
            check_folder(model_path)
            PATH = os.path.join(model_path, "dqn_agent_epi_" + str(cur_epi) + '.pth')
            torch.save(agent_dqn, PATH)

            # Save csv files of actions at each checkpoint episode
            act_path = os.path.join(output_folder, "action_checkpoints")
            check_folder(act_path)
            file_name = os.path.join(act_path, "dqn_act_epi_" + str(cur_epi) + '.csv')
            logging.info("Saving action of episode {} to {}".format(cur_epi, file_name))
            if not os.path.exists(file_name):
                pd.DataFrame(act_data).to_csv(file_name, header=['v', 'w'], index=False)
        writer.flush()  # flush to disk every episode
    
    logging.info("Plotting final figures")
    plt.figure()
    plt.title("DQN Training Loss")
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.plot(dqn_loss_x_axis, dqn_loss_list)
    plt.savefig(os.path.join(output_folder, "dqn_loss.png"))

    plt.figure()
    plt.title("Training Episode Reward")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(epi_r_x_list, train_r_list)
    plt.savefig(os.path.join(output_folder, "epi_reward.png"))

    if config['run_eval_in_rl']:
        plt.figure()
        plt.title("Evaluation Reward of Each Checkpoint")
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(reward_x_axis, reward_list)
        plt.savefig(os.path.join(output_folder, "eval_reward.png"))
    

    logging.info("Total time elapsed: {} mins".format(int(time.time()-init_time)/60))
    logging.info("Total episodes: {}".format(cur_epi))
    writer.close()


if __name__ == '__main__':
    now = datetime.now()
    current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    output_folder = os.path.join('./rl_model', current_time)
    check_folder(output_folder)
    log_name = os.path.join(output_folder, "train.log")
    level = logging.INFO
    format = '%(message)s'
    handlers = [logging.FileHandler(log_name), logging.StreamHandler()]
    # noinspection PyArgumentList
    logging.basicConfig(level=level, format=format, handlers=handlers)

    writer_dir = os.path.join(output_folder, "runs")
    writer = SummaryWriter(log_dir=writer_dir)
    command = "tensorboard --logdir=" + writer_dir
    # subprocess.run(["gnome-terminal", "-x", "sh", "-c", command])
    main(output_folder, writer)