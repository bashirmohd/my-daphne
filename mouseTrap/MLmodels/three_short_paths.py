#calculate three shortest paths and produce 24 values for bar graph
# 
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

import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

#from ml_predictor import simpleLSTM
#global values:
pos = {}

# UPDATE these to take input from GUI
src="SUNN"
dest="CHIC"
filesize=10

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
    
    return first_path,second_path, third_path

    


def main():
    #run ML model
    #store all 24 values in a array

    with open('predictions/sarima_predictions.json') as json_file:
        pred_24=json.load(json_file)

    #draw graph and hightlight three paths
    #get SRC and Dest
    print("Source: ")
    print(src)
    print("Destination: ")
    print(dest)
    print("File size (GB):")
    print(filesize)

    #returns three short paths:
    firstpath, secondpath, thirdpath =google_map()
    print("following paths returned")
    print(firstpath)

    #loop through each path and calculate the total for 24 values
    firstpath24values=np.zeros(24)
    n_first=len(firstpath)

    for j in range(n_first):
        for k in range(0,n_first-j-1):
            l1=firstpath[k]
            l2=firstpath[k+1]
            #get 24 values for each pair
            for p in pred_24['barpredictions']:
                if p['sc']==l1 and p['dst']==l2:
                    for a in range(24):
                        firstpath24values[a]+=p['sarima'][a]
                    print("onelink added")
                    print(l1)
                    print(l2)
                    print(firstpath24values)
            
            print(firstpath24values)





main()