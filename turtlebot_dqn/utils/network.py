import numpy as np
import torch
import torch.nn.functional as F


class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc(x))
        return self.fc2(x)


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, 
                gamma, target_update, reward_clip, use_soft_update, tau):
        self.action_dim = action_dim
        self.dqn_type = "DoubleDQN"

        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).cuda()
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).cuda()

        self.reward_clip = reward_clip
        self.reward_clipping_factor = 100  # max possible reward

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma  # discount factor
        self.target_update = target_update  # target-network update frequency
        self.count = 0
        self.tau = tau
        self.use_soft_update = use_soft_update

    def take_action_from_net(self, state):
        state = torch.tensor([state], dtype=torch.float).cuda()
        action = self.q_net(state).argmax().item()
        return action
    
    def get_value_function(self, state):
        state = torch.tensor([state], dtype=torch.float).cuda()
        max_action = self.q_net(state).max(1)[1].view(-1, 1)
        value = self.q_net(state).gather(1, max_action)
        return value 

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float32).cuda()
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).cuda()
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).view(-1, 1).cuda()
        
        if self.reward_clip:
            rewards = rewards / self.reward_clipping_factor
    
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32).cuda()
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).view(-1, 1).cuda()

        # print("state:{} | action:{} | reward:{}  | next_states:{} | dones:{}".format(states, actions, rewards, next_states, dones))
        
        q_values = self.q_net(states).gather(1, actions)  # Q-value
        # Max Q value of next state
        max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
        max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD error target
        
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # MSE function
        self.optimizer.zero_grad()
        dqn_loss.backward()  # backward

        self.optimizer.step()
        if self.use_soft_update:
            for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            if self.count % self.target_update == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())  # update target-network
        self.count += 1
        return dqn_loss
