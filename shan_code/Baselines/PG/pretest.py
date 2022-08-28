import os
import gym
import csv
import torch
import random
import numpy as np
import MetaRL as metaRL
from Dijkstra import dijkstra



def main(env_name, seed, cuda):
    cuda = bool(cuda)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_name = 'cpu'
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        device_name = 'cuda'
    device = torch.device(device_name)
    print(device)
    env = gym.make(env_name)
    env.seed(seed)
    ob_space = env.observation_space
    ob_size = []
    for local_ob in ob_space:
        ob_size.append(local_ob.shape[0])
    action_space = env.action_space
    actions_size = []
    for action in action_space:
        actions_size.append(action.n)

    policies = []

    for index in range(len(ob_size)):
        file = os.getcwd() + "/pretrainedmodels/policy" + "{}".format(index) + ".pth"
        policy = torch.load(file, map_location=torch.device('cpu'))
        policy = policy.to(device)
        policies.append(policy)

    nodes = env.backend.nodes
    nodes_connected_links = env.backend.nodes_connected_links
    corrected_num = 0
    total_num = 0
    _ = env.reset()
    done = False
    while not done:

        actions = []
        for index, node in enumerate(env.backend.nodes):
            policy = policies[index]
            local_ob = []
            if len(env.backend.nodes_queues[node.name]) > 0:
                dst = env.backend.nodes_queues[node.name][0].destination
                local_ob.append(dst.index)
                action = dijkstra(nodes, nodes_connected_links, node.name, dst.name)
                actions.append(action)
            else:
                local_ob.append(-1)
                action = 0
                actions.append(action)
            for _ in range(ob_size[index] - 1):
                local_ob.append(0)
            local_ob = torch.tensor(local_ob).float()
            output = policy(local_ob)
            total_num += 1
            if action == output:
                corrected_num += 1
                # print("actions", actions)
        _, _, done, _ = env.step(actions)
    print("Test accruray:", corrected_num / total_num)
    print("pre testing done")


if __name__ == '__main__':
    main(env_name='Deeproute-stat-v0',
         seed=42,
         cuda=1)
