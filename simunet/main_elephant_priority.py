#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn import decomposition
from sklearn.cluster import KMeans
import os
import sys
import re
import time


import pandas as pd 
from sklearn import cluster, datasets


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
#from agents.router import Router
from agents.node import Site
from agents.node import Edge
from agents.traffic_generator import FlowTraffic


from RLtechniques.dqnclasses import ReplayMemory
from RLtechniques.dqnclasses import NeuralNetwork


#from mllibrary import step

import networkx as nx
import matplotlib.pyplot as plt

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import json
#from PIL import Image

import torch
import torch.nn as nn
#import torch.optim as optim
#i#mport torch.nn.functional as F
#import torchvision.transforms as T


topology_file = "topo.json"


def read_json_file(filename):
	with open(filename) as f:
		js_data=json.load(f)

		nodes = js_data['data']['mapTopology']['nodes']
		edges = js_data['data']['mapTopology']['edges']			
	return(nodes, edges)



Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))




def main():
	#Create agents

	#read topology file
	read_nodes, read_edges=read_json_file(topology_file)
	print(read_nodes)
	print(read_edges)
	print(len(read_nodes))
	print(len(read_edges))
	
	#initial agents
	i=0
	nodeList=[]
	edgeList=[]

	for n in read_nodes:
		site_location=Site(i, i, n['name'])
		i=i+1
		nodeList.append(site_location)

	i=0
	for e in read_edges:
		edge_detail=Edge(i, i, e['BW'], e['Lat'], 'SRC','DST',0,0)
		i=i+1
		edgeList.append(edge_detail)


	for n in nodeList:
		testn=n
		print(testn.name)

	#generate traffic data
	#20% elephant = {5-10 GB, latency 10-15 time steps}
	#80%mice ={0.5-1 GB, 1-3 time steps}

	flowname=""

	TRAININGITERATION=10

	EPISODE_NUMBER=1

	total_FLOWS=100
	total_elephant=20
	total_mice=80

	#1 gam finishes when 100 flows are sent to destination

	globalt=0

	#initialize 2D matrix
	#Demand_matrix=Current utilization matrix + incoming flow
	demand_matrix= []


	#for training the game agent to play the game multiple times
	for j in range(TRAININGITERATION):
		#one game iteration start to terminal state

		all_flows=[]
		print_throughput0=0
		print_throughput1=0
		print_throughput2=0
		print_throughput3=0

		#generate elephant or mice flow 100 flows
		for f in range(total_mice):
			flowname="mouse"
			flowsize=random.randint(1,2)
			flowdur=random.randint(1,3)
			print(flowname,flowsize,flowdur)
			flow_created=FlowTraffic(f, flowname, flowsize,flowdur, 0,0)
			all_flows.append(flow_created)

		for f in range(total_elephant):
			flowname="elephant"
			flowsize=random.randint(3,5)
			flowdur=random.randint(5,8)
			print(flowname,flowsize,flowdur)
			flow_created=FlowTraffic(f+80, flowname, flowsize,flowdur, 0,0)
			all_flows.append(flow_created)

		

			#EorM= random.randint(0,9)
			#if between 2-9 mouse else elephant
			#if(EorM>1):
				
				
		#print(flowname,flowsize,flowdur)
		    #add active flows to queue
		#flow_created=FlowTraffic(f, flowname, flowsize,flowdur, 0,0)
		#	all_flows.append(flow_created)
		print("flows created")
		print(len(all_flows))
		
		for fi in all_flows:
			print("Flow is %s, %s, %s, %s" %(fi.num, fi.name,fi.size,fi.dur))

		
		#replay_mem=[]
		#create utilization 2D matrix
		perc_eorm_matrix=np.zeros((4,2))
		avail_util_path=np.zeros([4])

		#next time step gives reward and information on allocat flow, next state is terminal if overflows,
		#  if flow allocated more than capacity, we get a -1, 
		# if still capacity ok +0.1
		avail_util_paths=np.zeros([4])
		avail_util_paths[0]=10
		avail_util_paths[1]=10
		avail_util_paths[2]=8
		avail_util_paths[3]=8

		penalty=0
		active_flows=[]
		reward=0
		#record action_taken
		rec_action_taken=np.zeros([4])
		rec_flows_chosen=np.zeros([2]) #0 for mouse 1 elephant
		rec_flows_done=np.zeros([2])

		#choose one flow
		fa=0


		for i in range(EPISODE_NUMBER):
			#allocate flow to network
			#get flow
			active_flows=[]
			reward=0

			for fa in all_flows:
				#get current utilization matrix
				fi=fa # flow to allocate first one
				

				
				# code do without RL- brute force into it
				#assumption we are allocating the full flow on one path (no splitting)
				
				
				#Choose random action
				#get available bw
				print("available paths")
				print(avail_util_paths) #allocate on first available path
				#path_chosen = random.randint(0,3)
				#take first flow
				if fa.num==0 or fa.num==9 or fa.num==19 or fa.num==29 or fa.num==39 or fa.num==49 or fa.num==59 or fa.num==69 or fa.num==79 or fa.num==89 or fa.num==99:
					print_throughput0=10-avail_util_paths[0]
					print_throughput1=10-avail_util_paths[1]
					print_throughput2=8-avail_util_paths[2]
					print_throughput3=8-avail_util_paths[3]

					print("throughputs: ")
					print(print_throughput0)
					print(print_throughput1)
					print(print_throughput2)
					print(print_throughput3)


				

				flag=0

				if flag==0:
					if avail_util_paths[0]>fi.size:
						avail_util_paths[0] -= fi.size
						flow_to_allocate=FlowTraffic(fi.num, fi.name,fi.size,fi.dur,0,fi.dur+1)
						active_flows.append(flow_to_allocate)
					
						#all_flows.remove(fi)
						#perc=fi.size/10

						rec_action_taken[0]+=1
						if fi.name=="mouse":
							rec_flows_chosen[0]+=1
							perc_eorm_matrix[0][0]+=1

						else:
							rec_flows_chosen[1]+=1
							perc_eorm_matrix[0][1]+=1
						flag=1
						#print("allocated 0 %s" %fi.size)

				if flag==0:
					if avail_util_paths[1]>fi.size:
						avail_util_paths[1]-=fi.size
						flow_to_allocate=FlowTraffic(fi.num, fi.name,fi.size,fi.dur,1,fi.dur+1)
						active_flows.append(flow_to_allocate)
						rec_action_taken[1]+=1
					
						#all_flows.remove(fi)
						perc=fi.size/10
						if fi.name=="mouse":
							rec_flows_chosen[0]+=1
							perc_eorm_matrix[1][0]+=1
						else:
							rec_flows_chosen[1]+=1
							perc_eorm_matrix[1][1]+=1
						flag=1
						#print("allocated 1 %s" %fi.size)

				
				if flag==0:
					if avail_util_paths[2]>fi.size:
						avail_util_paths[2]-=fi.size
						flow_to_allocate=FlowTraffic(fi.num, fi.name,fi.size,fi.dur,2,fi.dur+1)
						active_flows.append(flow_to_allocate)
						rec_action_taken[2]+=1
					
						#all_flows.remove(fi)
						perc=fi.size/8

						if fi.name=="mouse":
							rec_flows_chosen[0]+=1
							perc_eorm_matrix[2][0]+=1

						else:
							rec_flows_chosen[1]+=1
							perc_eorm_matrix[2][1]+=1
						flag=1
						#print("allocated 2 %s" %fi.size)


				if flag==0:
					if avail_util_paths[3]>fi.size:
						avail_util_paths[3]-=fi.size
						flow_to_allocate=FlowTraffic(fi.num, fi.name,fi.size,fi.dur,3,fi.dur+1)
						active_flows.append(flow_to_allocate)
						rec_action_taken[3]+=1
					
						#all_flows.remove(fi)
						perc=fi.size/8

						if fi.name=="mouse":
							rec_flows_chosen[0]+=1
							perc_eorm_matrix[3][0]+=1

						else:
							rec_flows_chosen[1]+=1
							perc_eorm_matrix[3][1]+=1
						#print("allocated 3 %s" %fi.size)



				

				#claculate reward if any flows finish
			 	#remove expire flows on path	
				for af in active_flows:
					#print("af.counter %s" %af.counter)
					#print("with size %s" %af.size)
					#print("with duration %s" %af.dur)
					af.counter-=1

					if af.counter<=0:
						lat=0
						if af.allocated_to==0 or af.allocated_to==3:
							lat=3
						if af.allocated_to==1 or af.allocated_to==2:
							lat=1
						#print("lat is %s" %lat)
						completion_time=af.dur+lat
						#print("completion time %s" %completion_time)

						slowness = af.dur/completion_time
						this_reward=slowness
						reward+=this_reward
						#print("flow removed with size %s" %af.size)
						#print("reward %s " %reward)
						#print("slowness %s" %slowness)

						avail_util_paths[af.allocated_to]= avail_util_paths[af.allocated_to]+af.size
						if af.name=="mouse":
							rec_flows_done[0]+=1
							#perc_eorm_matrix[af.allocated_to][0]-=1
						else:
							rec_flows_done[1]+=1
							#perc_eorm_matrix[af.allocated_to][1]-=1
						active_flows.remove(af)

			#end of flows loop

			print("****************For iteration number %s " %j)
			#print(i)
			for path in range(4):
				print("Percentage flows on path number %s" %path)
				print("mice are %s" %perc_eorm_matrix[path][0])
				print("elephant are %s " %perc_eorm_matrix[path][1])
			#print("Next Episode")
			#get reward
			
			print("total reward at end of 100 flows %s" %reward)
			#print("penalty %s" %penalty)
			print("Active flows at end of episode %s" %len(active_flows))
			print("Actions taken:")
			for rat in rec_action_taken:
				print(rat)
			print("flows allocated")
			for rfc in rec_flows_chosen:
				print(rfc)
			print("flows done")
			for rfd in rec_flows_done:
				print(rfd)


		print("NextGame")
		#print(j)
		
		#Use poisson to introduce elephant
		#nextTime=random.expovariate(0.2)
		#print(nextTime)
	#Run agent functions







main()

#notes How to build this code:
# flows start from source (denv) and go to dest(lbnl): possible paths: Denv-> lbnl, Denv->sacr->lbnl, denv->kans->lbnl. 
# if flow go through kans add lat +3, if flows go through sacr add lat +1
# objective is to transfer flow in least time Total time to completion = Job living time + latency on path
# slowness factor: Job should complete by, job actually finished by 
# finish as many of 100 jobs as possible by allocating them across all three paths?
# achieve high utilization by allocating all bandwidth effieciently to all jobs
#state 0 3 0 0 0 -> 0 5 0 0 0
#reward agent gets a +ve reward at minimal slowness time - highly desired behavior
#penalty agent allocates too many flows with less available capacity (allocated to bad combinations flows, not enough will cause loss)
#negative reward slight: if no flows completed every time step (slight, we want the agent to learn late rather than making fast bad decisions)

#state space
#all possible situations i can allocate on 4,4 matrix grid with possible links 5. All combinations of src, dest (1 possible destinations), 100 flows
# account for flow (elephant and mouse 2 kinds) being on the link, all possible states 100 x 5 x x2 x1 = 1000 states (continuous states)
# not discrete state space, many combinations of GB 10*100*max(elephnat)*max(mice)
# Agent will experience one of these 500 states and take an action
# Actions allocate to A(kans)=0, B(Sacr)=1, C(lbnl)=2, Do nothing=3 

#4D array
# elep, # mouse, availa cap on each path
#[1,[e,m,a]]
#[2,[e,m,a]]
#[3,[e,m,a]]
