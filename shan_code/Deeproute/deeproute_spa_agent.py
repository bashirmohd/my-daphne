#!/usr/bin/env python
# coding: utf-8

import random
import gym
import sys
import networkx as nx
import gym_deeproute_stat
from Dijkstra import dijkstra, dijkstra_lat

MAX_TICKS = 500
total_reward = 0

env = gym.make('Deeproute-stat-v0')
observation = env.reset()
# print('link latency')
# for link in env.backend.links:
    # print('link name', link.name)
    # print('link lat', link.lat)


for t in range (MAX_TICKS):
    # print('active flows')
    actions = []
    # for flow in env.backend.active_flows:
    #     print('flow link', flow.to_link.name)
    #     print('flow lat', flow.lat)
    #     print('flow to node', flow.to_node_name)
    #     print('flow destination', flow.destination.name)
    # print('wating flows')
    # print(len(env.backend.nodes))
    # node = env.backend.nodes[0]
    # for flow in env.backend.nodes_queues[node.name]:
        # print('queue length', len(env.backend.nodes_queues[node.name]))
        # print('flow lat', flow.lat)
        # print('flow size', flow.bw)
        # print('flow destination', flow.destination.name)

    for node in env.backend.nodes:
        # print('node name', node.name)
        if len(env.backend.nodes_queues[node.name]) > 0:
            # print('queueing length', len(env.backend.nodes_queues[node.name]))
            flow = env.backend.nodes_queues[node.name][0]
            # print('flow lat', flow.lat)
            # print('flow destination', flow.destination.name)
            nodes = env.backend.nodes
            links = env.backend.links
            nodes_connected_links = env.backend.nodes_connected_links
            links_avail = env.backend.links_avail
            action = dijkstra(nodes, links, nodes_connected_links, links_avail, node.name, flow.destination.name)
            # action = dijkstra_lat(nodes, links, nodes_connected_links, node.name, flow.destination.name)
            actions.append(action)
        else:
            actions.append(0)
    # print('actions')
    node = env.backend.nodes[0]
    to_link, to_node_name = env.backend.nodes_connected_links[node.name][actions[0]]
    # print('link size', to_link.bw)
    # for index, node in enumerate(env.backend.nodes):
    #     to_link, to_node_name = env.backend.nodes_connected_links[node.name][actions[index]]
        # print('to_link', to_link.name)
        # print('to node name', to_node_name)
    observation, reward, done, task = env.step(actions)
    total_reward += reward
    # print(reward)
print(env.backend._generated_flows)
print(env.backend._delivered_flows)

print("Average reward", total_reward / (t +1) )
print("Episode Finished  after {} timesteps".format(t+1))

env.cleanup()

