import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import matplotlib.pyplot as plt
from simu import Simu
from wind_turbine import Wind_turbine, Wind
from math_utils import wrap_to_m180_p180
import exputils as eu
import exputils.data.logging as log

def default_config():
    return eu.AttrDict(
        # Environment
        env = "Simu",
        # Agent
        agent = "DQN",
        # Training
        episodes = 1000,
        render = False,
        # Hyperparameters
        gamma = 0.99,
        lr = 5e-4,
        batch_size = 64,
        update_every = 4,
        # Replay buffer
        buffer_size = int(1e5),
        # Epsilon
        eps_start = 1.0,
        eps_end = 0.01,
        eps_decay = 0.995,
        # Model
        hidden_layers = [64, 64],
        # Misc
        seed = 0,
        threshold = 5,
    )
        

class Linear_Network():
    def __init__(self, input_size, output_size, lr):
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(output_size)

    def forward(self, x):
        return np.dot(x, self.weights) + self.bias

    def backward(self, x, y, y_pred):
        error = y - y_pred
        print(error.shape, x)
        self.weights -= self.lr * np.dot(x.T, error)
        self.bias -= self.lr * error


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size):
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)  
        self.size = 0
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.size += 1
        self.size = min(self.size, self.buffer_size)
    

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to('cpu')
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to('cpu')
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to('cpu')
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to('cpu')
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to('cpu')
        return (states, actions, rewards, next_states, dones)

class Agent():
    def __init__(self, state_size, action_size, buffer_size, batch_size, tau, lr, gamma, update_every):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.tau = tau
        self.lr = lr
        self.gamma = gamma
        self.update_every = update_every
        self.buffer_size = buffer_size
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size).to('cpu')
        self.qnetwork_target = QNetwork(state_size, action_size).to('cpu')
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1)
        if (self.t_step % self.update_every) == 0:
            # checking if the memory has enough experience
            if self.memory.size > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
    
    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to('cpu')
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
 
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
# create a basic agent, which choose action based on just the state and a threshold
class BasicAgent():
    def __init__(self, state_size, action_size, threshold):
        self.state_size = state_size
        self.action_size = action_size
        self.threshold = threshold

    def act(self, state, eps=0.):
        state = wrap_to_m180_p180(state)
        if np.abs(state) - self.threshold < 0:
            return 1
        else:
            return np.sign(state) + 1
    
    def step(self, state, action, reward, next_state, done):
        pass


def train(env, agent, episodes = None, render = False):
    scores = []
    scores_window = deque(maxlen=100)
    eps = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    for i_episode in range(1, episodes+1):
        state = env.reset()
        score = 0
        while True:
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        print('\rEpisode {}\tScore: {:.2f}'.format(i_episode, score))
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        log.add_scalar("Average_Score", np.mean(scores_window))
        log.add_scalar("Score", score)
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
    return scores




def run(config):
    config = eu.combine_dicts(config, default_config())
    wind = Wind(10, 270, 3600, 'OU')
    wind_turbine = Wind_turbine(350, False)
    env = Simu(None, wind_model=wind, wind_turbine_model=wind_turbine, max_steps=int(np.ceil(7*24*3600/wind.step_duration) + 1))
    if config.agent == "basic":
        agent = BasicAgent(env.state_size, env.action_size, config.threshold)
    elif config.agent == "dqn":
        agent = Agent(1, 3, config.buffer_size, config.batch_size, config.tau, config.lr, config.gamma, config.update_every)
    #agent = BasicAgent(1, 3, 5)
    train(env, agent, 500, render=False)
