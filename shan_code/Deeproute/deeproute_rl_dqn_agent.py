#!/usr/bin/env python
# coding: utf-8

import random
import gym
import gym_deeproute_stat
import numpy as np
from collections import deque
from Model import Model
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import csv
import argparse
from torch.autograd import Variable

ENV_NAME = "Deeproute-stat-v0"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

# In[3]:

class DQNSolver:

    def __init__(self, observation_space, actions_space, hidden_shape, hidden_num):
        self.exploration_rate = EXPLORATION_MAX
        self.actions_space = actions_space
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.Models = []
        self.optimizers = []
        for index in range(len(actions_space)):
            temp = Model(observation_space, self.actions_space[index] + 1, hidden_shape, hidden_num)
            temp.to(self.device)
            self.Models.append(temp)
            self.optimizers.append(optim.SGD(temp.parameters(), lr = LEARNING_RATE, momentum = 0.9))
        self.criterion = nn.MSELoss()
        
    def remember(self, state, actions, reward, next_state):
        self.memory.append((state, actions, reward, next_state))

    def act(self, state):
        current_actions = []  
        if np.random.rand() < self.exploration_rate:
            for index in range(len(self.actions_space)):
                current_actions.append(random.randint(0, self.actions_space[index]))
            print ("Taking random action", current_actions)
            # return current_actions
        else:
            if self.device.type == 'cuda':
                state = Variable(state.cuda())
               
            for index in range(len(self.actions_space)):
                state.to(self.device)
                q_values = self.Models[index](state)
                _, action = torch.max(q_values, 1)
                current_actions.append(torch.Tensor.cpu(action).numpy()[0])
            print ("Taking predicted  action", current_actions)

        return current_actions

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, actions, reward, state_next in batch:
            if self.device.type == 'cuda':
                state = Variable(state.cuda())
                state_next = Variable(state_next.cuda())
            for index in range(len(actions)):
                q_update = (reward + GAMMA * torch.max(self.Models[index](state_next)))
                self.optimizers[index].zero_grad()
                # print(state.is_cuda)
                q_values = self.Models[index](state)
                q_values[0][actions[index]] = q_update
                if self.device.type == 'cuda':
                    q_values = Variable(q_values.cuda())

                loss = self.criterion(self.Models[index](state), q_values)
                loss.backward()
                self.optimizers[index].step()

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    # def save_model(self, model_name='Models.h5'):
    #     # serialize model to JSON
    #     models_json = self.Models.to_json()
    #     with open("models.json", "w") as json_file:
    #         json_file.write(models_json)
    #     # serialize weights to HDF5
    #     self.Model.save_weights(model_name)
    #     print("Saved model to disk")

def main(args):
    random.seed(100)
    env = gym.make(ENV_NAME)
    #score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space
    actions_space = []
    for action in action_space:
        actions_space.append(action.n)
    env.max_ticks  = args.n_max_ticks
    # print(actions_space)
    dqn_solver = DQNSolver(observation_space, actions_space, 300, 5) # add 300, 5 to arg...
    run = 0
    MAX_RUN = args.n_episodes
    score_card = []
    while run < MAX_RUN:
        run += 1
        state = env.reset()
        step = 0
        score = 0
        while True:
            step += 1
            state = torch.Tensor(state)
            actions = dqn_solver.act(state)
            state_next, reward, terminal = env.step(actions)
            score += reward
            state_next = torch.Tensor(state_next)
            dqn_solver.remember(state, actions, reward, state_next)
            state = state_next
            if terminal:
                print ("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(score))
                score_card.append((run, score, step))
                break
            dqn_solver.experience_replay()


    # plt.plot
    with open('dqn_stat_score_card_{0}.csv'.format(MAX_RUN), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(score_card)

    # dqn_solver.save_model('dqn_stat_model_{0}_run.h5'.format(MAX_RUN))


    env.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DQN Agent')
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=100)
    parser.add_argument(
        '--n-max-ticks',
        type=int,
        default=3000)
    
    args = parser.parse_args()

    main(args)

