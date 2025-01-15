from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!


# Pour ce projet on va réutiliser le DQN du notebook 4, en ajoutant quelques couches et en modifiant quelques paramètres
# imports
import torch
import torch.nn as nn
import numpy as np
import random

import os
from copy import deepcopy
import evaluate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



   

# Agent

class ProjectAgent:
    # Inner Class ReplayBuffer
    class ReplayBuffer:
        def __init__(self, capacity, device):
            self.capacity = int(capacity) # capacity of the buffer
            self.data = []
            self.index = 0 # index of the next cell to be filled
            self.device = device
        def append(self, s, a, r, s_, d):
            if len(self.data) < self.capacity:
                self.data.append(None)
            self.data[self.index] = (s, a, r, s_, d)
            self.index = (self.index + 1) % self.capacity
        def sample(self, batch_size):
            batch = random.sample(self.data, batch_size)
            return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
        def __len__(self):
            return len(self.data)


    # init -- no arg according to main.py
    def __init__(self):
        # device = "cuda" if next(model.parameters()).is_cuda else "cpu"

        # DQN
        state_dim = env.observation_space.shape[0]
        self.nb_actions = env.action_space.n 
        nb_neurons = 256
        self.model = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                                nn.ReLU(),
                                nn.Linear(nb_neurons, nb_neurons),
                                nn.ReLU(), 
                                nn.Linear(nb_neurons, nb_neurons), # added layer 
                                nn.ReLU(), 
                                nn.Linear(nb_neurons, nb_neurons), # added layer
                                nn.ReLU(), 
                                nn.Linear(nb_neurons, nb_neurons), # added layer
                                nn.ReLU(), 
                                nn.Linear(nb_neurons, self.nb_actions)).to(device)
        # target model
        self.target_model = deepcopy(self.model).to(device)
        self.update_target_freq = 300

        # monitoring
        self.monitoring_nb_trials = 5

        # buffer
        buffer_size = 1e6
        self.memory = self.ReplayBuffer(buffer_size, device)

        # greedy strategy
        self.epsilon_max = 1.0
        self.epsilon_min = 0.01
        self.epsilon_stop = 40000
        self.epsilon_delay = 200
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop


        self.gamma = 0.99      
        
        # training 
        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.nb_gradient_steps = 2
        self.batch_size = 1024
    
    # Greedy Action
    def greedy_action(self, state):
        # device = "cuda" if next(network.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = self.model(torch.Tensor(state).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            #update = torch.addcmul(R, self.gamma, 1-D, QYmax)
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, env, max_episode=200, save_path=f"{os.getcwd()}/mva_rl_weights.pth", verbose=False):
        if not self.model.training:
            self.model.train()
    
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0

        best_val_cum_reward = 0

        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample() # exploration
            else:
                action = self.greedy_action(state) # exploitation

            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # train
            for _ in range(self.nb_gradient_steps):
                self.gradient_step()

            # update target network - replace
            if step % self.update_target_freq == 0: 
                self.target_model.load_state_dict(self.model.state_dict())

            # next transition
            step += 1
            if done or trunc:
                episode += 1
                val_cum_reward = evaluate.evaluate_HIV(agent=self, nb_episode=self.monitoring_nb_trials)
                if verbose:
                    print("Episode ", '{:3d}'.format(episode), 
                        ", epsilon ", '{:6.2f}'.format(epsilon), 
                        ", batch size ", '{:5d}'.format(len(self.memory)), 
                        ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                        ", val ", '{:.2e}'.format(val_cum_reward),
                        sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0


                # save the best model (on evaluation)
                if val_cum_reward > best_val_cum_reward:
                    self.best_model = deepcopy(self.model) #initialisé au premier passage
                    self.save(save_path) # précaution
                    best_val_cum_reward = val_cum_reward # ajustement

            else:
                state = next_state
        self.model.load_state_dict(self.best_model.state_dict())
        self.save(save_path)
        return episode_return


    # necessary functions for the agent
    def act(self, observation, use_random=False):
        if use_random: # exploration
            return env.action_space.sample()
        with torch.no_grad(): # exploitation (greedy)
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        path_to_weights = f"{os.getcwd()}/mva_rl_weights.pth"
        self.model.load_state_dict(torch.load(path_to_weights, map_location=device))
        self.model.to(device)
        self.model.eval()