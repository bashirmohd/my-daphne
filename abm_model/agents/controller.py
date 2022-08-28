import json
import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))



class SimpleController(object):

	
	'''initialize memory of agent'''
	def __init__(self, num, temperature, src_name, dest_name, bandwidth_capacity):
		self.num=num
		self.temperature = temperature
		self.src_name=src_name
		self.dest_name= dest_name
		self.bandwidth_capacity=bandwidth_capacity
		print("created router")

class RyuController(object):
