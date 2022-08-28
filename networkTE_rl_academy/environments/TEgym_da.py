"""
This QT gym uses Continuous observation but a DISCRETE Action space
"""
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np



#MK: to do read yaml file and set up gym
  ################
    #47% efficiency in State of art

    #set up action
    # Qubit  - inductance, energy (-/+)
    #Resonators - LC , CC values  (-/+)
    #Eigenfrequencies (-/+)
    #loss-rate (-/+)
    #Anharmonicities (-/+)
    #Cross-kerr (-/+)
  ##########

    
#WAN TE Gym
class WANTEGymEnv(gym.Env):

  """
    Description:
       WAN TE Gym reads Yaml and sets up SDN actions
    Source:
       Adapted from cartpole Gym
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Microwave Power           -4.8                    4.8
        1       Microwave Frequency            -Inf                    Inf
        2       Optic Power           -4.8                    4.8
        3       Optic Frequency            -Inf                    Inf
    
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push Qubit inductance to the left
        1     Push Qubit inductance to the right
        Note: This needs to be changed to continous later
    Reward:
        Reward is 1 if cross Ker at step n is more than step n-1
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Episode length is greater than 200
        Solved Requirements:
        Considered solved to be added LATER
    """


  metadata = {'render.modes': ['human','rgb_array'],
   'video.frames_per_second': 50
   }

  def __init__(self):
    self.noise=0.1
    #ask Meriam to define any other env setting not part of learning
    self.move_mag = 10.0

    self.tau = 0.02  # seconds between state updates

    # Power of microwave to restart the experiment
    self.mwp_threshold=2.4
    self.pp_threshold=2
    self.prev_crossker=0.0 # to save cross ker from previous state
    
    high=np.array([self.mwp_threshold*2,
                  np.finfo(np.float32).max,
                  self.pp_threshold*2,
                  np.finfo(np.float32).max],
                  dtype=np.float32)

    self.action_space=spaces.Discrete(2)
    self.observation_space=spaces.Box(-high, high, dtype=np.float32)

    self.seed()
    self.viewer=None
    self.state=None
    self.qres=Q_ckt_coupled_qubit_resonance.QR()
    self.steps_beyond_done=None
    self.indthreshold=1



  def seed(self,seed=None):
    self.np_random,seed=seeding.np_random(seed)
    return[seed]

  def step(self,actionList):
    #print("ActionList")
    #print(actionList)
    if actionList==0:
      action = 0 # ++
    else:
      action = 1 # --
    err_msg = "%r (%s) invalid" % (action, type(action))
    assert self.action_space.contains(action), err_msg
    
    newlr=self.qres.return_current_Lj()
    reward=0

    if action == 0:
      newlr =newlr+0.1e-9
      #print("action ++")

    if action == 1:
      newlr = newlr-0.1e-9
      #print("action --")
      #print(newlr)
       
    
    done = bool(
        0.1e-9 > newlr or newlr > 100e-9
        #or theta < -self.theta_threshold_radians
        #or theta > self.theta_threshold_radians
    )
    #print(newlr)
    #print(done)
    self.qres.set_current_lj(newlr)

    efreq=self.qres.return_eigen_frequencies(newlr)
      #how much to move left or right
      #print("Efreq")
      #print(efreq)
    diff=efreq[1]- efreq[0]
    #print(diff)


    mp, mf, op, of = self.state
      #self.prev_crossker=crossker
    self.state = (mp, mf, op, of)

    reward = diff

    if not done:
      reward=diff
      
    
    #elif self.steps_beyond_done is None:
      # Pole just fell!
      # self.steps_beyond_done = 0
    #  reward = -1e-15
    else:
      if self.steps_beyond_done == 0:
        logger.warn(
          "You are calling 'step()' even though this "
          "environment has already returned done = True. You "
          "should always call 'reset()' once you receive 'done = "
          "True' -- any further steps are undefined behavior."
          )
        self.steps_beyond_done += 1
        reward = -1e-15
        #self.reset()
    #print(efreq)
    print(reward)
    return np.array(self.state), reward, done, {}


  def reset(self):
    #reset at random values in state, we could start from zero 
    self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
    self.steps_beyond_done = None
    return np.array(self.state)


  def render(self, mode='human'):
    print("todo rendering graphs")
    
  def close(self):
    if self.viewer:
      self.viewer.close()
      self.viewer = None
