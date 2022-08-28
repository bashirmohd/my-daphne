#!/usr/bin/env python
# coding: utf-8

import random
import gym
import gym_deeproute_stat
import numpy as np
from collections import deque
from Model import model
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from torch.autograd import Variable

MEMORY_SIZE = 50000

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.0
EXPLORATION_DECAY = 0.995

EPOCHES = 100


class DQNSolver:

    def __init__(self, observation_space, action_space, hidden_shape, hidden_num, lr, device):
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = action_space
        self.device = device
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.Model = model(observation_space, self.action_space, hidden_shape, hidden_num, self.device)
        self.Model = self.Model.to(self.device)
        self.optimizer = optim.SGD(self.Model.parameters(), lr=lr, momentum=0.9)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, flag):
        self.memory.append((state, action, reward, next_state, flag))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action = random.randint(0, self.action_space - 1)

        else:
            q_values = self.Model(state)
            action = torch.argmax(q_values)
            action = action.item()

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

        return action

    def sample_batch(self, bsz):
        if len(self.memory) < bsz:
            return []

        batch = random.sample(self.memory, bsz)

        return batch

    def experience_replay(self, state, action, q_update):

        self.optimizer.zero_grad()
        q_values = self.Model(state)
        q_values[action] = q_update

        if self.device.type == 'cuda':
            q_values = Variable(q_values.cuda())
        output = self.Model(state)
        loss = self.criterion(output, q_values)
        loss.backward()
        self.optimizer.step()


def main(env_name, lr, n_episodes, layer_shape, layer_num, bsz, cuda):
    cuda = bool(cuda)
    device_name = 'cpu'

    if cuda and torch.cuda.is_available():
        device_name = 'cuda'

    device = torch.device(device_name)
    print(device)

    random.seed(100)

    env = gym.make(env_name)
    observation_space = env.observation_space
    action_space = env.action_space
    actions_space = []
    for action in action_space:
        actions_space.append(action.n)
    observations_space = []
    for ob_size in observation_space:
        observations_space.append(ob_size.shape[0])

    nodes_connected_links = env.backend.nodes_connected_links
    nodes = env.backend.nodes

    dqn_solvers = {}
    for index, ob_size in enumerate(observations_space):
        node = nodes[index]
        dqn_solver = DQNSolver(ob_size, actions_space[index], layer_shape, layer_num, lr, device)
        dqn_solvers[node.name] = dqn_solver

    q_delivery_time_record = []
    q_packet_loss_record = []

    run = 0
    MAX_RUN = n_episodes
    while run < MAX_RUN:
        run += 1
        observations = env.reset()
        step = 0
        while True:
            step += 1
            actions = []
            for index, observation in enumerate(observations):
                dqn_solver = dqn_solvers[nodes[index].name]
                action = dqn_solver.act(observation)
                actions.append(action)
            next_observations, rewards, terminal, flags = env.step(actions)

            for index, action in enumerate(actions):
                next_ob = next_observations[index]
                state = observations[index]
                reward = rewards[index]
                flag = flags[index]
                dqn_solvers[nodes[index].name].remember(state, action, reward, next_ob, flag)

            observations = next_observations

            if terminal:
                global_packet_loss, global_average_delivery_time = env.get_packet_loss_and_delivery_time()
                q_delivery_time_record.append(global_average_delivery_time)
                q_packet_loss_record.append(global_packet_loss)
                print("run", run)
                print('packet loss', global_packet_loss)
                print('average_delivery_time', global_average_delivery_time)
                print("generated_packets:", env.backend.generated_packets)
                print("delivered_packets:", env.backend.delivered_packets)
                break
        for _ in range(EPOCHES):
            for index, node in enumerate(nodes):
                dqn_solver = dqn_solvers[node.name]
                batch = dqn_solver.sample_batch(bsz)
                for state, action, reward, next_state, flag in batch:
                    _, next_node_name = nodes_connected_links[node.name][action]
                    next_model = dqn_solvers[next_node_name].Model
                    next_q_values = next_model(state)
                    if flag == 1:
                        tau = 0
                        q_update = reward + tau
                    else:
                        tau = torch.max(next_q_values)
                        q_update = reward + tau.item()

                    dqn_solver.experience_replay(state, action, q_update)

        if run != 0 and run % 10 == 0:
            with open('results.csv', 'w') as csvfile:
                resultswriter = csv.writer(csvfile, dialect='excel')
                resultswriter.writerow(["name", "value"])
                resultswriter.writerow(["flow", env.flow_lambda])
                resultswriter.writerow(["topology", env.get_task()])
                resultswriter.writerow(["packet_loss", q_packet_loss_record])
                resultswriter.writerow(["completion_time", q_delivery_time_record])

    env.cleanup()

    with open('results.csv', 'w') as csvfile:
        resultswriter = csv.writer(csvfile, dialect='excel')
        resultswriter.writerow(["name", "value"])
        resultswriter.writerow(["flow", env.flow_lambda])
        resultswriter.writerow(["topology", env.get_task()])
        resultswriter.writerow(["packet_loss", q_packet_loss_record])
        resultswriter.writerow(["completion_time", q_delivery_time_record])


if __name__ == '__main__':

    main(env_name="Deeproute-stat-v0",
         lr=1e-6,
         n_episodes=500,
         layer_shape=200,
         layer_num=3,
         bsz=30,
         cuda=1)


