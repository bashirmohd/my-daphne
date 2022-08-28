import json
import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))



class FlowTraffic(object):

	
	'''initialize memory of agent'''
	def __init__(self, num, name):
		self.num=num
		self.name=name
		print("New flow created with num: %d " % num)