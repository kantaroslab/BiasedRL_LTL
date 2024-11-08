import random
import numpy as np
import collections
import heapq
import os
import networkx as nx
import time
import logging
import glob
import shutil
from LTL2DRA.dra_utils import check_sys2ap_rabin_ros


def shortestPath(edges, source, sink):
    # https://gist.github.com/hanfang/89d38425699484cd3da80ca086d78129
    graph = collections.defaultdict(list)
    for l, r, c in edges:
        graph[l].append((c, r))
    # create a priority queue and hash set to store visited nodes
    queue, visited = [(0, source, [source])], set()
    heapq.heapify(queue)
    cnt = 0
    while queue:
        (cost, node, path) = heapq.heappop(queue)
        if node not in visited:
            if cnt != 0:
                visited.add(node)
                path = path + [node]
                if node == sink:
                    return path
            cnt += 1
            for c, neighbour in graph[node]:
                if neighbour not in visited:
                    heapq.heappush(queue, (cost + c, neighbour, path))
    return None


class Rabin_Automaton_ROS(object):
    def __init__(self, ltl, coord_dict):
        # shortest paths for each dra state
        self.distance_map = {}
        # pruned dra_state - ap dictionary
        self.processed_dra = {}

        self.if_global = False

        # logging.info("\n[Initializing Rabin_Automaton_ROS]")
        logging.info("LTL: {}".format(ltl))
        self.ltl = ltl
        self.coord_dict = coord_dict
        with open("command.ltl", mode='w') as f:
            f.write(ltl)
        result1 = os.system("./ltlfilt -l -F \"command.ltl\" --output=\"command.ltl\"")
        # Update: remove --stutter=no to make sure the translation is equal to HOA format for ./ltl2dstar
        # result2 = os.system(
        #     "./ltl2dstar --ltl2nba=spin:ltl2ba --stutter=no --output-format=dot command.ltl command.dot")
        result2 = os.system("./ltl2dstar --ltl2nba=spin:ltl2ba --output-format=dot command.ltl command.dot")
        result3 = os.system("./ltlfilt --lbt-input -F \"command.ltl\" --output=\"command_read.ltl\"")
        # result4 = os.system("dot -Tpdf command.dot > command.pdf")
        result5 = os.system("./ltl2dstar command.ltl command.txt")

        if result1 == 0:
            logging.info("LTL Filtering Succeeded!")
            os.system("cat command.ltl")
        if result2 == 0:
            logging.info("DRA Dot Graph Generation Succeeded!")
            # os.system("cat command.dot")
        if result3 == 0:
            logging.info("LTL Translated into readable form in command_read.ltl")
        #     os.system("cat command_read.ltl")
        # if result4 == 0:
        #     logging.info("DRA Pdf file generated into command.pdf")
        if result5 == 0:
            # Extract the number of accepting pairs information from the HOA file
            # Source: https://www.ltl2dstar.de/docs/ltl2dstar.html#output-format-hoa
            logging.info("HOA file generated successfully.")
            with open("command.txt") as f:
                lines = f.readlines()
            info = ""
            for line in lines:
                line_ = line.replace("\n", "")
                if "Acceptance-Pairs" in line_:
                    info = int(line_.replace("Acceptance-Pairs: ", ""))
                    break
            self.num_of_accepting_pairs = info
            logging.info("Number of accepting pairs is: {}".format(self.num_of_accepting_pairs))

        rabin_graph = nx.nx_agraph.read_dot("command.dot")
        rabin_graph.remove_nodes_from(["comment", "type"])

        self.graph = rabin_graph
        self.num_of_nodes = len(self.graph.node)

        self.accept = [int(i) for i in self.graph.node if "+0" in self.graph.node[i]["label"]]
        self.reject = [int(i) for i in self.graph.node if "-0" in self.graph.node[i]["label"]]

        self.accepting_pair = {'L': self.accept, 'U': self.reject}

        logging.info("accept: {}".format(self.accept))
        logging.info("reject: {}".format(self.reject))

        logging.info("Accepting pair: {}".format(self.accepting_pair))

        self.deadlock = []
        for i in self.reject:
            if str(i) in self.graph[str(i)].keys():
                if " true" in [self.graph[str(i)][str(i)][j]["label"]
                               for j in range(len(self.graph[str(i)][str(i)]))]:
                    self.deadlock.append(i)
        logging.info("deadlock: {}".format(self.deadlock))
        for i in self.graph.node:
            if "fillcolor" in self.graph.node[i].keys():
                if self.graph.node[i]["fillcolor"] == "grey":
                    self.init_state = int(i)
                    break
        logging.info("initial: {}".format([self.init_state]))

    def get_graph(self):
        return self.graph

    def get_init_state(self):
        return self.init_state

    def check_current_ap(self, coord):
        # a system state can only be in 1 AP at a certain time step (correct)
        res_ap = []
        ans = check_sys2ap_rabin_ros(coord, self.coord_dict)
        if ans is not None:
            res_ap.append(ans)
        # print("Atomic Proposition satisfied by {} -> {}".format(coord, res_ap))
        return res_ap

    def prune_dra(self, state_aps):
        # Edges will be pruned as long as there are more than 1 positive AP that need to be satisfied
        pruned_aps = []
        for ap in state_aps:
            pos, neg = seperate_ap_sentence(ap)
            if len(pos) > 1:
                continue
            pruned_aps.append(ap)
        return pruned_aps

    def dra_distance_dict_generation(self):
        # Set up the distance function which minimize the distance from current DRA state to the goal
        # since the DRA state is named based on increasing number
        # try to find the shortest path towards the biggest number
        # print([p for p in nx.all_shortest_paths(self.graph, source='1', target='5')])
        # print(self.graph.node)
        # Data structure:
        # (head, tail): {path}

        """
        Update: redo the pruning function
        For []<>e1&&[]<>e2&&[]!e4&&[]!e5
        The path <4, 4> should be <4, 2, 4>
        Check why it fails
        """

        for start_node in self.graph.node:
            tmp_dict = {}
            for end_node in self.graph[start_node]:
                for k in range(len(self.graph[start_node][end_node])):
                    ap = self.graph[start_node][end_node][k]['label']
                    pos, _ = seperate_ap_sentence(ap)
                    if len(pos) <= 1:
                        if end_node not in tmp_dict:
                            tmp_dict[end_node] = []
                        tmp_dict[end_node].append(self.graph[start_node][end_node][k]['label'])
            self.processed_dra[start_node] = tmp_dict
        # for k, v in self.processed_dra.items():
        #     print(k, v)
        # print(self.processed_dra)
        print("DRA pruning completed")

        edges = []
        for start_node in self.processed_dra.keys():
            for end_node in self.processed_dra[start_node].keys():
                edges.append((start_node, end_node, 1))

        for start_node in self.graph.node:
            for end_node in self.graph.node:
                path = shortestPath(edges, start_node, end_node)
                if path is not None:
                    self.distance_map[(path[0], path[-1])] = path

        # for k, v in self.distance_map.items():
        #     print(k, v)
        # print(self.distance_map)
        print("Distance map generation completed")
        print("DRA automaton setup completed.")

    def select_next_rabin(self, source, rabin, goal_list, goal_size, ap_range_map):
        # source is the mixture state.
        source_rabin = str(source[-1])
        target_rabin = None
        if len(rabin.accept) == 1:
            target_rabin = str(rabin.accept[0])  # assert accept rabin state is only 1
        else:
            for target_rabin in rabin.accept:
                target_rabin = str(target_rabin)
                if (source_rabin, target_rabin) in self.distance_map:
                    res_path = self.distance_map[(source_rabin, target_rabin)]
                    if len(res_path) == 1:
                        next_rabin = res_path[0]
                    else:
                        next_rabin = res_path[1]
                    if next_rabin in self.processed_dra[source_rabin]:
                        break
        res_path = self.distance_map[(source_rabin, target_rabin)]
        # print("selecting rabin from {} to {} | Path: {}".format(source_rabin, target_rabin, res_path))
        next_rabin = res_path[1]
        # if len(res_path) == 1:
        #     next_rabin = res_path[0]
        # else:
        #     next_rabin = res_path[1]
        # logging.info("Next objective rabin is: {}".format(next_rabin))
        # print("Possible choice: {}".format(self.processed_dra[source_rabin][next_rabin]))
        if len(self.processed_dra[source_rabin][next_rabin]) == 1:
            new_goal_ap_sentence = self.processed_dra[source_rabin][next_rabin][0]
            pos, neg = seperate_ap_sentence(new_goal_ap_sentence)
        else:
            # Multiple accepting states
            # new_goal_ap_sentence = random.choice(self.processed_dra[source_rabin][next_rabin])
            new_goal_ap_sentence = self.processed_dra[source_rabin][next_rabin][0]
            pos, neg = seperate_ap_sentence(new_goal_ap_sentence)
            if len(pos) == 0:
                # Update: pick the closest to the current position
                # no positive APs -> select randomly from grid world
                dis = float("inf")
                choice = None
                for i in range(len(goal_list)):
                    l = np.sqrt((goal_list[i][0] - goal_list[i][0]) ** 2 + (goal_list[i][1] - goal_list[i][1]) ** 2)
                    if l < dis:
                        dis = l
                        choice = i
                # logging.info("No active AP sentence to serve as goal, choosing nearest grid in workspace")
                # cur_goal = random.choice(goal_list)
                cur_goal = goal_list[choice]
                c = goal_size / 2
                gx, gy = cur_goal[0] + c, cur_goal[1] + c

                # TODO: check this section

                return (gx, gy, goal_size), "Any"
        # logging.info("New AP sentence is: {}".format(new_goal_ap_sentence))
        return ap_range_map[str(pos[0])], str(pos[0])

    def next_state(self, current_state, next_coord):
        ap_next = self.check_current_ap(next_coord)
        next_states = self.possible_states(current_state[-1])
        for i in next_states:
            next_state_aps = self.processed_dra[str(current_state[-1])][str(i)]
            if " true" in next_state_aps:
                return current_state[-1]
            else:
                for j in next_state_aps:
                    if self.check_ap(ap_next, j):
                        return i

    def possible_states(self, current_rabin_state):
        return [int(i) for i in self.processed_dra[str(current_rabin_state)].keys()]

    def check_ap(self, ap_next, ap_sentence):
        # print("ap_next->{} | ap_sentence->{}".format(ap_next, ap_sentence))
        pos, neg = seperate_ap_sentence(ap_sentence)
        # print("pos->{} | neg->{}".format(pos, neg))
        if set(ap_next).issuperset(set(pos)) and self.check_neg(ap_next, neg):
            """
            If ap_next satisfies:
                1) ap_next is the superset of the current positive APs in the sentence
                2) ap_next does not falls into any of the negative APs 
            Then we choose this ap_sentence as our next AP_sentence 
            """
            return True
        return False

    def check_neg(self, ap, negs):
        for i in ap:
            if i in negs:
                return False
        return True


def seperate_ap_sentence(input_str):
    return_str = []
    if len(input_str) > 1:
        index = find_ampersand(input_str)
        if len(index) >= 1:
            return_str = [input_str[0:index[0]]]
        else:
            return_str = input_str
            if '!' in return_str:
                return [], [return_str.replace('!', '')]
            else:
                return [return_str], []
        for i in range(1, len(index)):
            return_str += [input_str[index[i - 1] + 1:index[i]]]
        return_str = return_str + [input_str[index[-1] + 1:]]
        return_str = [i.replace(' ', '') for i in return_str]
    elif len(input_str) == 1:
        return_str = input_str
    elif len(input_str) == 0:
        raise AttributeError('input_str has no length')

    without_negs = []
    negations = []
    for i in range(len(return_str)):
        if '!' in return_str[i]:
            negations += [return_str[i].replace('!', '')]
        else:
            without_negs += [return_str[i]]
    return without_negs, negations


def find_ampersand(input_str):
    index = []
    original_length = len(input_str)
    original_str = input_str
    while input_str.find('&') >= 0:
        index += [input_str.index('&') - len(input_str) + original_length]
        input_str = original_str[index[-1] + 1:]
    return index
