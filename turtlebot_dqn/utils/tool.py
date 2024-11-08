import random
import collections
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def plot_weighted_graph(G, name):
    # pos = {(x, y): (y, -x) for x, y in G.nodes()}
    plt.figure(figsize=(20, 20))
    pos = {(x, y): (x, y) for x, y in G.nodes()}
    nx.draw(G, pos=pos,
            node_color='lightgreen',
            with_labels=True,
            node_size=400)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.savefig(name)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # setup max memory

    def add(self, state, action, reward, next_state, done):
        # At each time step, store transition to D
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


class Dynamics:
    def __init__(
            self,
            At,
            bt,
            ct,
            u_limits=None,
            dt=1.0,
            c=None,
            sensor_noise=None,
            process_noise=None,
    ):
        # State dynamics
        self.At = At
        self.bt = bt
        self.ct = ct
        self.num_states, self.num_inputs = bt.shape
        # Observation Dynamics and Noise
        if c is None:
            c = np.eye(self.num_states)
        self.c = c
        self.num_outputs = self.c.shape[0]
        self.sensor_noise = sensor_noise
        self.process_noise = process_noise

        # Min/max control inputs
        self.u_limits = u_limits

        self.dt = dt
