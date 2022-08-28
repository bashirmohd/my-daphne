import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from gym import Env


class Link():
    """ Source node"""
    #metadata={"render.modes":["human"]}

    def __init__(self,id, src, dest, capacity):
        self.id=id
        self.src=src
        self.dest=dest
        self.capacity=capacity
        #print("link added")

   
   