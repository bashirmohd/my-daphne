# univariate multi-step encoder-decoder lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Activation, Dropout
import os
import sys
import re
import time
import json

import matplotlib.pyplot as plt
import networkx as nx

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

#global values:

pos = {}

def json_to_dict(filename, graph_feature):
    json_data = open(filename)
    data = json.load(json_data)
    #get the list of dicts
    dicts = data['data']['mapTopology'][graph_feature]
    return dicts
 
def build_graph(nodes, edges):
    G = nx.Graph()
    for node in nodes:
        site = node['name']
        position = (node['x'], node['y'])
        G.add_node(site, pos=position)
    for edge in edges:
        node1 = edge['ends'][0]['name']
        node2 = edge['ends'][1]['name']
        G.add_edge(node1, node2)
    return G

def fill_pos(nodes):
    for node in nodes:
        site = node['name']
        position = (node['x'], node['y'])
        pos[site] = position

def main():
    #read in map topology
    #build ES.net map
    node_dicts = json_to_dict('../topology_files/esnet_nodes.json', 'nodes')
    edge_dicts = json_to_dict('../topology_files/esnet_edges.json', 'edges')

    G = build_graph(node_dicts, edge_dicts)
    pos=fill_pos(node_dicts)
    nx.draw(G, nx.get_node_attributes(G, 'pos'), with_labels=True)
    plt.show()
    
    #get src and dest

    #Calculate all possible paths between src and dest


main()