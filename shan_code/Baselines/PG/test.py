import os
import csv
import gym
import torch
import random
import numpy as np
import cherry as ch
import MetaRL as metaRL
from copy import deepcopy
from policies import Policy

from cherry.algorithms import a2c, trpo
from cherry.models.robotics import LinearValue

from torch import autograd
from torch.nn.utils import parameters_to_vector, vector_to_parameters





def compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states):

    returns = ch.td.discount(gamma, rewards, dones)
    baseline.fit(states, returns)
    values = baseline(states)
    next_values = baseline(next_states)
    bootstraps = values * (1.0 - dones) + next_values * dones
    next_value = torch.zeros(1, device=values.device)
    return ch.pg.generalized_advantage(tau=tau,
                                       gamma=gamma,
                                       rewards=rewards,
                                       dones=dones,
                                       values=bootstraps,
                                       next_value=next_value)


def maml_loss(train_episodes, learner, baseline, gamma, tau, device):
    states = train_episodes.state()
    actions = train_episodes.action()
    rewards = train_episodes.reward()
    dones = train_episodes.done()
    next_states = train_episodes.next_state()
    states = states.to(device, non_blocking=True)
    actions = actions.to(device, non_blocking=True)
    rewards = rewards.to(device, non_blocking=True)
    dones = dones.to(device, non_blocking=True)
    next_states = next_states.to(device, non_blocking=True)
    log_probs = learner.log_prob(states, actions)
    advantages = compute_advantages(baseline, tau, gamma, rewards,
                                    dones, states, next_states)
    advantages = ch.normalize(advantages).detach()
    return a2c.policy_loss(log_probs, advantages)


def fast_adapt(clone, train_episodes, adapt_lr, baseline, gamma, tau, device):
    loss = maml_loss(train_episodes, clone, baseline, gamma, tau, device)
    gradient = autograd.grad(loss, clone.parameters(), retain_graph=True, create_graph=True)

    return metaRL.algorithms.maml.maml_update(clone, adapt_lr, gradient)


def main(env_name,adapt_lr,adapt_steps,adapt_bsz,tau,gamma,seed,cuda):
    
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
    env_ticks = env.max_ticks
    
    env.seed(seed)
    '''
    Get the observation size and action size. Now, the global observation space is shared by all the routers, so the observation size is       the same for all the routers. However, each router has a different action size which equals to the number of neighbor routers.
    
    ''' 
    ob_space = env.observation_space
    ob_size = []
    for local_ob in ob_space:
        ob_size.append(local_ob.shape[0])
    
    ### Each router has an individual policy neural network
    policies = []

    for index in range(len(ob_size)):
        file = os.getcwd() + "/savemodels/policy" + "{}".format(index) + ".pth"
        policy = torch.load(file)
        policy = policy.to(device)
        policies.append(policy)

        
    ### Current baseline: Benchmarking Deep Reinforcement Learning for Continuous Control
    baselines = []
    for local_ob_size in ob_size:
        baseline = LinearValue(local_ob_size)
        baseline = baseline.to(device)
        baselines.append(baseline)
        
    reward_record = []
    task_files = ["simple.json", "topo2.json", "topo0.json"]
    
    tasks = [{'topo_file': file} for file in task_files]
    for task_config in tasks:  # Samples batch of tasks
        ### clones of the current policies are used to sample trajectories
        task_reward = 0.0
        clones = []
        for policy in policies:
            clones.append(deepcopy(policy))
        ### The topology of the network might change after setting a new task
        env.set_task(task_config)
        task = metaRL.sample_trajectory(env)
        ### get the initialization of a specific task
        for step in range(adapt_steps):
            train_episodes, _ = task.run(clones, episodes=adapt_bsz)

            for index, clone in enumerate(clones):
                clone = fast_adapt(clone, train_episodes[index], adapt_lr, baselines[index], gamma, tau, device)
                
        valid_episodes, global_reward = task.run(clones, episodes=adapt_bsz)
            
            
        task_reward += sum(global_reward) / adapt_bsz
        reward_record.append(task_reward)
    
        # Print statistics
        print('\nTask', task_config['topo_file'])
        print('task_reward', task_reward)
        global_flow_loss, global_average_delivery_time = env.get_flow_loss_and_delivery_time()
        print('packet loss', global_flow_loss)
        print('average_delivery_time', global_average_delivery_time)
        
    
    # save results
    
    with open('test_results.csv', 'w') as csvfile:
        resultswriter = csv.writer(csvfile, dialect='excel')
        resultswriter.writerow(["adapt_lr",adapt_lr])
        resultswriter.writerow(["adapt_bsz",adapt_bsz])
        resultswriter.writerow(["reward_record",reward_record])
                    


if __name__ == '__main__':
    main(env_name='Deeproute-stat-v0',
        adapt_lr=0.005,
        adapt_steps=3,
        adapt_bsz=10,
        tau=1.00,
        gamma=1.00,
        seed=42,
        cuda=1)

