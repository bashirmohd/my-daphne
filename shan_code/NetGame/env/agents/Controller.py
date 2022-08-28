#routint table
#sender

#Traffic is generated here
#uses TCP algorithms to send data

import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from gym import Env

####
"""
Path 0=t/100
Path 1=45
Path 2=45
Path 3=1/100
"""
####

class Controller():
    """ Source node"""
    #metadata={"render.modes":["human"]}

    def __init__(self,id):
        self.id=id
        self.graph=None
        #self.pathtaken=pathtaken
        #print("controllerhere")

    def save_topology(self, graph):
        self.graph=graph

    def get_topology(self):
        return self.graph

    

    

