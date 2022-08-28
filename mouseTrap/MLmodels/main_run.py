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
import numpy as np

import matplotlib.pyplot as plt
import networkx as nx

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from ml_predictor import simpleLSTM
#global values:
pos = {}
src="SUNN"
dest="CHIC"

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

def machine_learning_main():
    ml_predictor()



def google_map():
    #read in map topology
    #build ES.net map
    node_dicts = json_to_dict('../topology_files/esnet_nodes.json', 'nodes')
    edge_dicts = json_to_dict('../topology_files/esnet_edges.json', 'edges')

    G = build_graph(node_dicts, edge_dicts)
    pos=fill_pos(node_dicts)
    #UNCOMMENT following 2 lines
    #nx.draw(G, nx.get_node_attributes(G, 'pos'), with_labels=True)
    #plt.show()
    
    #get src and dest
    paths=nx.shortest_simple_paths(G,source=src, target=dest)
    ct=0
    #for pt in paths:
    #    print(pt)
     #   print(np.size(pt))
     #   ct=ct+1
    #print(ct)
    # select the three least hop paths
    print("Printing Size")
    
    #find the shortest path
    print("shortest paths are")
    paths_list=list(paths)
    print(len(paths_list))

    first_path=paths_list[0]
    print(first_path)
    second_path=paths_list[1]
    print(second_path)
    third_path=paths_list[2]
    print(third_path)
    

    #print total prediction on three paths
    n_first=len(first_path)
    for j in range(0,n_first-1):
        print(first_path[j])
        print(first_path[j+1])
        #get 24 values for each pair

    ---> read in values from table

    for i in range(24):
        totalrow1=row1+row1
        
    
    #print(len(first_path))


def main():
    #run ML model
    #store all 24 values in a array

    #draw graph and hightlight three paths
    google_map()

main()