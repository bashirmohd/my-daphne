import json
import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
#implementing everything in keras due to pytorch dependency issues: https://deepsense.ai/keras-or-pytorch/


######################################################################
# Replay Memory
# -------------
#
# We'll be using experience replay memory for training our DQN. It stores
# the transitions that the agent observes, allowing us to reuse this data
# later. By sampling from it randomly, the transitions that build up a
# batch are decorrelated. It has been shown that this greatly stabilizes
# and improves the DQN training procedure.
#
# For this, we're going to need two classses:
#
# -  ``Transition`` - a named tuple representing a single transition in
#    our environment. It essentially maps (state, action) pairs
#    to their (next_state, reward) result, with the state being the
#    screen difference image as described later on.
# -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
#    transitions observed recently. It also implements a ``.sample()``
#    method for selecting a random batch of transitions for training.
#


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class NeuralNetwork(nn.Module):

    def __init__(self, num_layers=1, hidden_size=12):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = 4
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 10
        self.replay_memory_size = 10000
        self.minibatch_size = 30

        self.input_size = 5
        self.hidden_size = hidden_size
        self.output_size = 4

        self.num_layers = num_layers

        self.layers = torch.nn.ModuleList([
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(inplace=True),
        ])

        for i in range(self.num_layers):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.layers.append(nn.ReLU(inplace=True))

        self.layers.append(nn.Linear(self.hidden_size, self.output_size))

        # self.conv1 = nn.Conv2d(4, 30, 8, 4)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(32, 64, 4, 2)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.conv3 = nn.Conv2d(64, 64, 3, 1)
        # self.relu3 = nn.ReLU(inplace=True)
        # self.fc4 = nn.Linear(3136, 512)
        # self.relu4 = nn.ReLU(inplace=True)
        # self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        # out = self.conv1(x)
        # out = self.relu1(out)
        # out = self.conv2(out)
        # out = self.relu2(out)
        # out = self.conv3(out)
        # out = self.relu3(out)
        # out = out.view(out.size()[0], -1)
        # out = self.fc4(out)
        # out = self.relu4(out)
        # out = self.fc5(out)

        return output


class Qtable(nn.Module):
    def __init__(self, path0, path1,path2, path3, mOrE, valueQ, act, m0,e0,m1,e1,m2,e2,m3,e3):
        self.path0=path0
        self.path1=path1
        self.path2=path2
        self.path3=path3
        self.mOrE=mOrE
        self.act=act
        self.valueQ=valueQ
        self.m0=m0
        self.e0=e0
        self.m1=m1
        self.e1=e1
        self.m2=m2
        self.e2=e2
        self.m3=m3
        self.e3=e3

class GameState:
    def __init__(self, path0, path1,path2, path3, mOrE,act):
        self.path0=path0
        self.path1=path1
        self.path2=path2
        self.path3=path3
        self.mOrE=mOrE
        self.act=act


            