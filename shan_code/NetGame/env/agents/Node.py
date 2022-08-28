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

class Node():
    """ Source node"""
    #metadata={"render.modes":["human"]}

    def __init__(self, id):
        self.id=id
        #self.pathtaken=pathtaken
        #print("here")

    #def generate_traffic(self, cars, case, all_paths_available):
     #   s=np.random.poisson(5,cars) # 20 traffic generated
      #  print("2")
        #attach random routes

        #if random:
        #get all possible paths

    def message_imhere(self):
        #self.pathtaken=pathtaken
        print("I am here with id ", self.id)
        
               

       

