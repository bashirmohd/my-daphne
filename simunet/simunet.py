import json
import networkx as nx
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import matplotlib.animation

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

pos = {}
big_path = ['PNWG', 'SACR', 'SUNN', 'ELPA', 'HOUS', 'NASH', 'ATLA']
big_path_2 = ['DENV', 'KANS', 'CHIC', 'WASH', 'CERN-513', 'CERN-773', 'LOND', 'AMST']

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def json_to_dict(filename, graph_feature):
    json_data = open(filename)
    data = json.load(json_data)
    #get the list of dicts
    dicts = data['data']['mapTopology'][graph_feature]
    return dicts

def build_graph(nodes, edges):
    G = nx.Graph()
    for node in node_dicts:
        site = node['name']
        position = (node['x'], node['y'])
        G.add_node(site, pos=position)
    for edge in edge_dicts:
        node1 = edge['ends'][0]['name']
        node2 = edge['ends'][1]['name']
        G.add_edge(node1, node2)
    return G

def fill_pos(nodes):
    for node in node_dicts:
        site = node['name']
        position = (node['x'], node['y'])
        pos[site] = position

#send a flow across 2 links
#start, mid, end-- refer to the nodes of the path
def flow(num):
    ax.clear()
    path = big_path[:num+2]
    path_2 = big_path_2[:num+2]
    nx.draw(G, nx.get_node_attributes(G, 'pos'), with_labels=True)

    #a path
    query_nodes = nx.draw_networkx_nodes(G, pos=pos, nodelist=path, node_color="blue", ax=ax)
    query_nodes.set_edgecolor("white")
    nx.draw_networkx_labels(G, pos=pos, labels=dict(zip(path,path)), ax=ax)
    edgelist = [path[k:k+2] for k in range(len(path) - 1)]
    nx.draw_networkx_edges(G, pos=pos, edgelist=edgelist, edge_color="blue", ax=ax)

    #another path
    query_nodes = nx.draw_networkx_nodes(G, pos=pos, nodelist=path_2, node_color="green", ax=ax)
    query_nodes.set_edgecolor("white")
    nx.draw_networkx_labels(G, pos=pos, labels=dict(zip(path_2,path_2)), ax=ax)
    edgelist = [path_2[k:k+2] for k in range(len(path_2) - 1)]
    nx.draw_networkx_edges(G, pos=pos, edgelist=edgelist, edge_color="green", ax=ax)

    # Scale plot ax
    ax.set_title("Frame %d:    "%(num+1) +  " - ".join(path), fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def traffic_generator():
    #simulate incoming flows
    #create flow

    #for i
    print("nothing")



def RL_agent():
    #at source Attach RL
    print("Nothing2")




    
fig, ax = plt.subplots(figsize=(15,10))
node_dicts = json_to_dict('esnet_nodes.json', 'nodes')
edge_dicts = json_to_dict('esnet_edges.json', 'edges')
#build ES.net map
G = build_graph(node_dicts, edge_dicts)
fill_pos(node_dicts)

ani = matplotlib.animation.FuncAnimation(fig, flow, frames=6, interval=500, repeat=False)
plt.show()

