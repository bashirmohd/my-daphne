#!/usr/bin/env python
# coding: utf-8

import random
import gym
import sys
import networkx as nx
import gym_deeproute_stat
from Dijkstra import dijkstra
import matplotlib.pyplot as plt
import numpy as np

num_iteration = 50
spa_delivery_time_record = []
q_delivery_time_record = []
random_delivery_time_record = []
q_packet_loss_record = []
spa_packet_loss_record = []

env = gym.make('Deeproute-stat-v0')

MAX_TICKS = env.max_ticks

# Q routing
observation = env.reset()
for task in env.sample_tasks(num_iteration):
    print(task)
    observation = env.reset()
    env.re_count()
    done = False
    while not done:
        actions = []
        for index, node in enumerate(env.backend.nodes):
            dst_name = node.name
            dst_index = observation[index]
            for node1 in env.backend.nodes:
                if node1.index == dst_index[0]:
                    dst_name = node1.name
            Q_values = env.backend.nodes_Q[node.name][dst_name]
            action = np.argmin(Q_values)
            actions.append(action)

        observation, _, done, task = env.step(actions)

    global_packet_loss, global_average_delivery_time = env.get_packet_loss_and_delivery_time()
    q_delivery_time_record.append(global_average_delivery_time)
    q_packet_loss_record.append(global_packet_loss)
    print('packet loss', global_packet_loss)
    print('average_delivery_time', global_average_delivery_time)
    print("generated_packets:", env.backend.generated_packets)
    print("delivered_packets:", env.backend.delivered_packets)

    print("Episode Finished  after {} timesteps".format(t+1))
#
# random action
# observation = env.reset()
# for task in env.sample_tasks(num_iteration):
#     print(task)
#     env.set_task(task)
#     for t in range(MAX_TICKS):
#         actions = env.action_space.sample()
#         observation, _, done, task = env.step(actions)
#
#     global_packet_loss, global_average_delivery_time = env.get_packet_loss_and_delivery_time()
#     random_delivery_time_record.append(global_average_delivery_time)
#     print('packet loss', global_packet_loss)
#     print('average_delivery_time', global_average_delivery_time)
#     print("generated_packets:", env.backend.generated_packets)
#     print("delivered_packets:", env.backend.delivered_packets)
#     print("Episode Finished  after {} timesteps".format(t+1))

fig, ax = plt.subplots()
ax.plot(q_delivery_time_record, 'b', label='Q')
ax.set_ylabel("Average packet completion time")
ax.set_xlabel("Iterations")
plt.title("inflow rate = {}".format(env.backend.flow_lambda[1]))
leg = ax.legend()
plt.grid()

fig, ax = plt.subplots()
ax.plot(q_packet_loss_record, 'b', label='Q')
ax.set_ylabel("packet loss")
ax.set_xlabel("Iterations")
plt.title("inflow rate = {}".format(env.backend.flow_lambda[1]))
leg = ax.legend()


plt.grid()
plt.show()

