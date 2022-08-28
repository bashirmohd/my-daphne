import json
import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))



class Site(object):

	
	'''initialize memory of agent'''
	def __init__(self, num, siteID, name):
		self.num=num
		self.siteID = siteID
		self.name=name
		print("New Site Node created with name:")
		print(name)

class Edge(object):

	'''initialize memory of agent'''
	def __init__(self, num, edgeID, bw, lat, node1, node2):
		self.num=num
		self.edgeID = edgeID
		self.bw=bw
		self.lat=lat
		self.node1=node1
		self.node2=node2
		print("New Edge created with id: %d" % edgeId)

