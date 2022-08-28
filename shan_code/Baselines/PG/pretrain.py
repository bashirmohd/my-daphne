import os
import gym
import csv
import torch
import random
import numpy as np
import MetaRL as metaRL
from policies import Policy
from Dijkstra import dijkstra

from torch import autograd



def main(env_name, learning_rate, hidden_size, hidden_layers, num_iter, seed, cuda):
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

    # Each router has an individual policy neural network
    policies = []

    for index, act_size in enumerate(actions_size):
        policy = Policy(ob_size[index], act_size, hidden_size, hidden_layers, device=device)
        policy = policy.to(device)
        policies.append(policy)

    nodes = env.backend.nodes
    nodes_connected_links = env.backend.nodes_connected_links
    for step in range(num_iter):
        print(step)
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
                action = torch.tensor([[action]]).float()
                loss = -policy.log_prob(local_ob, action)
                print(loss)
                gradient = autograd.grad(loss, policy.parameters(), retain_graph=True, create_graph=True)
                policy = metaRL.algorithms.maml.maml_update(policy, learning_rate, gradient)

            _, _, done, _ = env.step(actions)
            
    print("pre training done")

    # save model
    for index, policy in enumerate(policies):
        file = os.getcwd() + "/pretrainedmodels/policy" + "{}".format(index) + ".pth"
        torch.save(policy, file)


if __name__ == '__main__':
    main(env_name='Deeproute-stat-v0',
         learning_rate=0.001,
         hidden_size=200,
         hidden_layers=3,
         num_iter=3,
         seed=42,
         cuda=1)
