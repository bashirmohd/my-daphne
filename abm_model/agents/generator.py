import json
import os
import sys
import argparse
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))



class TaskGenerator(object):

	
	'''initialize memory of agent'''
	def __init__(self, num, tasks):
		self.num=num
		self.tasks = tasks
	

	def generateTask(num):
		random.randint(1,5)





