#!/usr/bin/env python3

import cherry as ch
import numpy as np
import torch


class sample_trajectory():

    def __init__(self, env):
        self.env = env
        self._needs_reset = True
        self._current_state = None
        
    def reset(self, *args, **kwargs):
        self._current_state = self.env.reset(*args, **kwargs)
        self._needs_reset = False
        return self._current_state

    def run(self,get_action,episodes):
        """
        Runner wrapper's run method.
        """
        replays = [ch.ExperienceReplay() for _ in range(len(get_action))]
        global_reward = []
        packet_loss = []
        delivery_time = []

        collected_episodes = 0
        
        while True:
            if collected_episodes >= episodes:
                self._needs_reset = True
                return replays, global_reward, packet_loss, delivery_time
            if self._needs_reset:
                self.reset()
            info = {}
            actions = []
            for index, policy in enumerate(get_action):
                action = policy(torch.Tensor(self._current_state[index]))
                actions.append(action)

            old_state = self._current_state
            actions =[action.item() for action in actions]
            state, rewards, done, _ = self.env.step(actions)
            # print(rewards)

            if done:
                collected_episodes += 1
                self._needs_reset = True
                pl, ct = self.env.get_packet_loss_and_delivery_time()
                packet_loss.append(pl)
                delivery_time.append(ct)
                
            for index in range(len(replays)):
                replays[index].append(torch.Tensor(old_state[index]), 
                                      torch.Tensor([actions[index]]), 
                                      rewards[index], 
                                      torch.Tensor(state[index]), 
                                      done, 
                                      **info)
            # print(rewards)
            global_reward.append(rewards[-1])
           
            self._current_state = state
            

