import gym
import json
import datetime as dt

#On your terminal run  pip install stable-baselines

from env.agents import Node
from env.agents import Controller
from env.agents import Link

#from braess_paradox_env.agents import EnvironmentAgent

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import json

""" 
agent-based model of a simple simulation sending flows from src , dest)
"""

"""
forward-class 0  Shortest possible route IGP
forward-class 1  Min delay route
forward-class 2  congestion-aware routing / minimize possible loss
forward-class 3  A* dynamic routing? 
forward-class 4  auto steering for on-demand hops ?
forward-class 5
"""


traffic_dict={'flows':[
                {'source':1,'sink':4,'flowsize':5,'forward-class':0},
                {'source':2,'sink':4,'flowsize':5,'forward-class':0}
            ]}

def iteration_starts(all_nodes, all_links, ControllerA):
    #discover my topology
    print("here")
    #tell controller my nodes and links


    G=nx.DiGraph()

   
    for i in all_nodes:
        i.message_imhere()
        G.add_node(i.id,data=i)
    #build the graph in controller memory

    for l in all_links:
        G.add_edge(l.src,l.dest, weight=2)
   

    pos=nx.circular_layout(G) # positions for all nodes

    nx.draw(G,pos, with_labels=True,font_weight='bold')
    plt.show()
    ControllerA.save_topology(G)




def main():
    # start the day
    #Declare all agents

    ControllerA=Controller.Controller(1)
    #4 nodes
    NodeA=Node.Node(1)
    NodeB=Node.Node(2)
    NodeC=Node.Node(3)
    NodeD=Node.Node(4)

    all_nodes=[]
    all_nodes.append(NodeA)
    all_nodes.append(NodeB)
    all_nodes.append(NodeC)
    all_nodes.append(NodeD)


    #links available
    LinkA=Link.Link(1,1,2,10)
    LinkB=Link.Link(2,2,4,10)
    LinkC=Link.Link(3,1,3,10)
    LinkD=Link.Link(4,3,4,10)

    all_links=[]
    all_links.append(LinkA)
    all_links.append(LinkB)
    all_links.append(LinkC)
    all_links.append(LinkD)


    #episode starts to 100 steps

    for epi in range(1):
        
        iteration_starts(all_nodes, all_links, ControllerA)
        #Controller has a map of network in its head
    
        #now start and send flows traffic
        traffic_flows=json.dumps(traffic_dict)

        #for t in traffic_flows['flows']:
        traffic_flows= json.loads(traffic_flows)
        #regular flows per episode
        for singleflow in traffic_flows['flows']:
            #print(singleflow['source'])
            #print("graph")
            H=nx.DiGraph()
            H=ControllerA.get_topology()
            #print(nx.nodes(H))
            print("shortest route is for single flow ", singleflow)
            print(nx.shortest_path(H,singleflow['source'],singleflow['sink']))

    
    #graph of time with flows
    #flow 
    
    #perflowSR
    #min_dleay route forward-class0
    #spfforward-class0
    #automated steeringforward-class0
    """
    print("back")

    G=nx.DiGraph()

    G.add_node(0,data=SourceA)
    G.add_node(1,data=LinkA)
    G.add_node(2,data=RouterA)
    G.add_node(3,data=LinkB)
    G.add_node(4,data=DestinationA)
    #G.add_node(2,data=DestinationA)

    G.add_edge(0,1, weight=2)
    G.add_edge(1,2, weight=2)
    G.add_edge(2,3, weight=2)
    G.add_edge(3,4, weight=2)

    mapping={0:'Source',2:'Router',4:'Sink'}
    H=nx.relabel_nodes(G,mapping)
    nx.draw(H, with_labels=True,font_weight='bold')
    plt.show()

    #Run simulation for 10 steps (1 sec = 1 steps)
    #speed 1 packet per step
    #source generates traffic?
    #ep=episode
    
    for ep in range(0,10):
        s=np.random.poisson(10) # 20 traffic generated
        print(s)


    #new=NodeA.generate_traffic(20) #generate 20 cars

    #print(new)
    #each car has a path to follow
    #selfish car - takes shortest path
    # random car (any path added)
    # if RL take cost to weight the path....

    #destination agent captures 100 cars (add cost) 
    #
    """



main()