import json
import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))



class VM(object):

	
	'''initialize memory of agent'''
	def __init__(self, num, userID, size, vm_monitor, task_schedular):
		self.num=num
		self.userID = userID
		self.size=size
		self.vm_monitor= vm_monitor
		self.task_schedular=task_schedular
		print("New VM created with id: %d" % num)

class DTN(object):

	'''initialize memory of agent'''
	def __init__(self, num, userID, size, vm_monitor, task_schedular):
		self.num=num
		self.temperature = temperature
