#Simple_topoRL

import json
import networkx as nx
import matplotlib as mpl
from networkx.readwrite import json_graph

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import matplotlib.animation

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T






# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


fn="topo.json"
def read_json_file(filename):
	with open(filename) as f:
		js_data=json.load(f)

		nodes = js_data['data']['mapTopology']['nodes']
		edges = js_data['data']['mapTopology']['edges']

		G = nx.Graph()
		#print(dicts)
		for node in nodes:
			site = node['name']
			print(site)

			position = (node['x'], node['y'])
			G.add_node(site)
		for edge in edges:
			edgename=edge['name']
			G.add_edge()
			
	return(G)

		#for edge in edge_dicts:
		#	node1 = edge['ends'][0]['name']
		#	node2 = edge['ends'][1]['name']
		#	G.add_edge(node1, node2)

def read_edge_data():
	return edges

 
def main():
	#build graph
	graph_read=read_json_file(fn)

	nx.draw_networkx_nodes(graph_read)

	print("here")


main()