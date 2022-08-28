import os
import time
import pylab
import random
import logging
import matplotlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors



class NODE(object):
    def __init__(self, name, posx, posy):
        self.name = name
        self.pos = (posx, posy)

class LINK(object):
    def __init__(self, name, bw, lat, node1, node2):
        self.name = name
        self.bw = bw
        self.lat = lat 
        self.node2 = node2
        self.node1 = node1

class FlowTraffic(object):
    def __init__(self, bw, dur, destination):
        self.bw = bw
        self.lat = dur
        self.flag = 0
        self.counter = dur
        self.to_link = None
        self.to_node_name = None
        self.destination = destination

class StatBackEnd(object):
    
    def __init__(self, flow_lambda, links, nodes, history, seed):
        np.random.seed(seed)
        self.nodes_queues = {}
        self.active_flows = []
        self._history = history
        self._loss_flows = 0
        self._delivered_flows = 0
        self._generated_flows = 0
        self.nodes_actions_history = {}
        self.flow_lambda = flow_lambda
        self.nodes = self.gen_nodes(nodes)
        self.links = self.gen_edges(links)
        self.ticks = [0] * len(self.nodes)
        self.links_avail = self.gen_links_avail()
        self._delivery_time_node = [0] * len(self.nodes)
        self._delivered_flows_node = [0] * len(self.nodes)
        self._delivery_time_local = [0] * len(self.nodes)
        self._delivered_flows_local = [0] * len(self.nodes)
       
        self.nodes_connected_links, self.nodes_connected_nodes = self.gen_nodes_connected_links()
        
    def gen_nodes_connected_links(self):
        nodes_connected_links = {}
        nodes_connected_nodes = {}
        for index1, node in enumerate(self.nodes):
            nodes_connected_links[node.name] = []
            nodes_connected_nodes[index1] = []
            for link in self.links:
                if link.node1 == node.name or link.node2 == node.name:
                    if link.node1 == node.name:
                        nodes_connected_links[node.name].append((link, link.node2))
                        for index2, connected_node in enumerate(self.nodes):
                            if connected_node.name == link.node2:
                                nodes_connected_nodes[index1].append(index2)
                    else:
                        nodes_connected_links[node.name].append((link, link.node1))
                        for index2, connected_node in enumerate(self.nodes):
                            if connected_node.name == link.node1:
                                nodes_connected_nodes[index1].append(index2)
        return nodes_connected_links, nodes_connected_nodes
                    
        
    def gen_edges(self,links):
        edgelist = []
        for e in links:
            edge_detail = LINK(e["name"], e["BW"], e["Lat"], e["from"], e["to"])
            edgelist.append(edge_detail)
        return edgelist
        
    def gen_nodes(self, nodes):
        nodeslist = []
        for n in nodes:
            node_detail = NODE(n["name"], n["posx"], n["posy"])
            nodeslist.append(node_detail)
        return nodeslist
        
    def gen_links_avail(self):
        links_avail = {}
        for link in self.links:
            links_avail[link.name] = link.bw
        return links_avail
        
            
    def generate_queues(self, node_index, node_name, reset = False, K = 1, Occur_pro = 1): 

        if reset:
            self.nodes_queues[node_name] = []

        if np.random.uniform(0,1) <= Occur_pro:
            self.ticks[node_index] = 0
            for _ in range(K):
                self._generated_flows += 1
                new_f_bw = np.random.poisson(self.flow_lambda[0])
                new_f_lat = 0
                new_f_destination = np.random.choice(self.nodes, 1, replace=False)
                while new_f_destination[0].name == node_name:
                    new_f_destination = np.random.choice(self.nodes, 1, replace=False)
                self.nodes_queues[node_name].append(FlowTraffic(new_f_bw, new_f_lat, new_f_destination[0]))
                

    def cleanup(self):
        pass
    
    def reset(self, links):
        self.active_flows.clear()
        self._delivery_time_node = [0] * len(self.nodes)
        self._delivered_flows_node = [0] * len(self.nodes)
        self._delivered_flows = 0
        self._generated_flows = 0
        self._loss_flows = 0
        self._delivery_time_local = [0] * len(self.nodes)
        self._delivered_flows_local = [0] * len(self.nodes)
       
        self.links = self.gen_edges(links)
        self.links_avail = self.gen_links_avail()
        
        for index, node in enumerate(self.nodes):
            
            self.generate_queues(index, node.name, reset = True, K = 3) ### initial flows at each waiting queue: 3
        
        for node in self.nodes:
            if node.name not in self.nodes_actions_history:
                self.nodes_actions_history[node.name] = [] 
            for index in range(self._history):
                new_f_bw = np.random.poisson(self.flow_lambda[0])
                new_f_lat = np.random.randint(1, 4)
                new_f_destination = np.random.choice(self.nodes, 1, replace=False)
                action = np.random.choice(np.arange(len(self.nodes_connected_links[node.name])), 1)
                self.nodes_actions_history[node.name].append(action[0])
                to_link, to_node_name = self.nodes_connected_links[node.name][action[0]]
                current_flow = FlowTraffic(new_f_bw, new_f_lat, new_f_destination[0])
                current_flow.counter = index + 1
                if self.links_avail[to_link.name] > new_f_bw:
                    self.links_avail[to_link.name] -= new_f_bw
                    current_flow.to_link = to_link
                    current_flow.lat += to_link.lat
                    current_flow.to_node_name = to_node_name
                    self.active_flows.append(current_flow)
            while len(self.nodes_actions_history[node.name]) < self._history:
                self.nodes_actions_history[node.name].append(-1)
  

    def take_actions(self, actions):
        
        for index in range(len(self.ticks)):
            self.ticks[index] += 1
        
        for flow in self.active_flows:
            flow.counter -= 1
        
        for node in self.nodes:
            for flow in self.nodes_queues[node.name]:
                flow.lat += 1
        
        for index, node in enumerate(self.nodes):
            occur_pro = 1 - np.exp(- self.ticks[index] * self.flow_lambda[1]) ### 1 - exp(- lambda t)
            # print(occur_pro)
            self.generate_queues(index, node.name, Occur_pro = occur_pro)

        for index, node in enumerate(self.nodes):
            
            if len(self.nodes_queues[node.name]) > 0:
                current_flow = self.nodes_queues[node.name][0]
                to_link, to_node_name = self.nodes_connected_links[node.name][actions[index]] 
                self.nodes_queues[node.name].remove(current_flow)
                self.nodes_actions_history[node.name].append(actions[index])
                if self.links_avail[to_link.name] > current_flow.bw:
                    current_flow.counter += to_link.lat
                    current_flow.lat += to_link.lat
                    current_flow.to_link = to_link
                    current_flow.to_node_name = to_node_name
                    self.links_avail[to_link.name] -= current_flow.bw
                    self.active_flows.append(current_flow)
                else:
                    self._loss_flows += 1
                    current_flow.flag = 1
                    current_flow.counter += to_link.lat
                    current_flow.lat += 50
                    current_flow.destination.name = to_node_name
                    current_flow.to_link = to_link
                    current_flow.to_node_name = to_node_name
                    self.active_flows.append(current_flow)
                    
            while len(self.nodes_actions_history[node.name]) > self._history:
                self.nodes_actions_history[node.name].pop(0)
        
        for flow in self.active_flows:
            # print(flow.to_link.name)
            if flow.counter <= 0.1:
                self.active_flows.remove(flow)
                if self.links_avail[flow.to_link.name] > 0:
                    self.links_avail[flow.to_link.name] += flow.bw
                if flow.to_node_name != flow.destination.name:
                    self.nodes_queues[flow.to_node_name].append(flow)
                else:
                    # print(flow.lat)
                    for index, node in enumerate(self.nodes):
                        if flow.to_node_name == node.name:
                            self._delivered_flows_local[index] += 1
                            self._delivery_time_local[index] += flow.lat
                            if flow.flag == 0:
                                self._delivery_time_node[index] += flow.lat
                                self._delivered_flows_node[index] += 1
                            break

    
    def render(self):
        ##### Network Figure
        fig = plt.figure()

        G = nx.Graph()
        for node in self.nodes:
            G.add_node(node.name)
            G.nodes[node.name]["pos"] = node.pos
            
        for link in self.links:
            G.add_edge(link.node1,link.node2)

        pos=nx.get_node_attributes(G,'pos')
        # pos = nx.spring_layout(G)
        nodes_labels = {}
        for node in G.nodes():
            nodes_labels[node] = node

        nodes = nx.draw_networkx_nodes(G, pos, node_size = 800)
        edges = nx.draw_networkx_edges(G, pos, width = 3)
        labels = nx.draw_networkx_labels(G, pos, labels = nodes_labels, font_size=18)
        nx.draw(G,pos)
        plt.savefig('topo.pdf')

        plt.show()


        