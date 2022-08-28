#dummy file
import random

from gym import spaces
import gym
import numpy as np


#1. __init__ function
#2. generate_network function
#3. get_state function
#4. reset function
#5. step function and get_reward function


class SDNController(gym.Env):
    """
    Define environment.

    The environment defines  how links will be selected based on bandwidth
    availability 
    """

    def __init__(self, max_ticks=10):

        # General variables defining the environment
        self.MAX_TICKS = max_ticks

        #PARAMETERS FOR EACH LINK
        self.link0_bw = 15  #bandwidth of LINK 0
        self.link0_pd = 5   #propagation delay of LINK 0
        self.link1_bw = 10
        self.link1_pd = 2
        self.link2_bw = 30
        self.link2_pd = 10

        #PARAMETERS TO GENERATE UDP TRAFFIC
        self.mu0 = 5.0
        self.sigma0 = 2.0
        self.mu1 = 4.0
        self.sigma1 = 1.0
        self.mu2 = 5.5
        self.sigma2 = 0.5          


        self.action_space = spaces.Discrete(3)  #select any of the 3 channels

        # Observation 
        low = np.array([0.0,	#minimum current bw of link 0
                        0.0,	#minimum available bw of link 0
                        0.0,	#minimum propagation delay of link 0
                        0.0,	#minimum current bw of link 1
                        0.0,	#minimum available bw of link 1
                        0.0,	#minimum propagation delay of link 1
                        0.0,	#minimum current bw of link 2
                        0.0,	#minimum available bw of link 2
                        0.0		#minimum propagation delay of link 2
                        ])
        high = np.array([self.link0_bw, self.link0_bw, self.link0_pd,
                        self.link1_bw, self.link1_bw, self.link1_pd,
                        self.link2_bw, self.link2_bw, self.link2_pd])

        self.observation_space = spaces.Box(low, high, dtype=np.float32)


    def generate_network(self):

        #self.active_link  = action
        """ Send udp traffic and then take bandwidth measurement """

        # send udp traffic  - simulate other flow    
        udp0_bw = np.random.normal(self.mu0, self.sigma0)   
        udp1_bw = np.random.normal(self.mu1, self.sigma1) 
        udp2_bw = np.random.normal(self.mu2, self.sigma2) 
        
        
        # measure link available bw
        self.available0_bw = max(0.0, float(self.link0_bw) - float(udp0_bw))
        self.available1_bw = max(0.0, float(self.link1_bw) - float(udp1_bw))
        self.available2_bw = max(0.0, float(self.link2_bw) - float(udp2_bw))
        

        # calculate current bw of each link
        self.current0_bw = np.random.normal(self.available0_bw, 0.5)
        self.current1_bw = np.random.normal(self.available1_bw, 0.5)
        self.current2_bw = np.random.normal(self.available2_bw, 0.5)


    def step(self, action):
        done = False
        self.ticks+=1
        reward = self.get_reward(action)


        self.generate_network()
        next_state = self.get_state()

        if self.ticks == self.MAX_TICKS:
            done = True
        return next_state, reward, done, {}


    def get_reward(self,action):

        if action == 0:
            if float(self.current0_bw) > float(self.available0_bw):  #if there is not enough bandwidth in link 0
                reward = -10
            else:	#if there is enough bandwidth in link 0
                reward = 10
            reward -= self.link0_pd	#give negative reward corresponding to the propagation delay

        elif action == 1:
            if float(self.current1_bw) > float(self.available1_bw):	#if there is not enough bandwidth in link 1
                reward = -10
            else:	#if there is enough bandwidth in link 1
                reward = 10
            reward -= self.link1_pd	#give negative reward corresponding to the propagation delay
            
        elif action == 2:
            if float(self.current2_bw) > float(self.available2_bw):	#if there is not enough bandwidth in link 2
                reward = -10
            else:	#if there is enough bandwidth in link 2
                reward = 10
            reward -= self.link2_pd	#give negative reward corresponding to the propagation delay   

        return reward


    def reset(self):
        self.ticks = 0
        self.generate_network()
        return self.get_state()


    def get_state(self):
        ob = np.array([self.current0_bw,  self.available0_bw, self.link0_pd,
                    self.current1_bw,  self.available1_bw, self.link1_pd,
                    self.current2_bw,  self.available2_bw, self.link2_pd])
        return ob

    def render(self, mode='human', close=False):
        return

