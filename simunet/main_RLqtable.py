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
from RLtechniques.dqnclasses import Qtable



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







def training(passed_q_table):
	#Create agents
	# #read topology file
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
	TRAININGITERATION=500
	EPISODE_NUMBER=1
	total_FLOWS=100
	#1 gam finishes when 100 flows are sent to destination
	# #initialize 2D matrix
	# #Demand_matrix=Current utilization matrix + incoming flow
	# #initialize Q table
	
	
	alpha=0.1
	gamma=0.6
	epsilon=0.1
	
	all_epochs=[]
	all_penalities=[]
	q_table=passed_q_table

	#for training the game agent to play the game multiple times
	for j in range(TRAININGITERATION):
		#one game iteration start to terminal state

		all_flows=[]
		

		#generate elephant or mice flow 100 flows
		for f in range(total_FLOWS):
			EorM= random.randint(0,9)
			#if between 2-9 mouse else elephant
			if(EorM>1):
				flowname="mouse"
				flowsize=random.randint(1,2)
				flowdur=random.randint(1,3)
			else:
				flowname="elephant"
				flowsize=random.randint(3,5)
				flowdur=random.randint(5,8)

			print(flowname,flowsize,flowdur)
			#add active flows to queue
			flow_created=FlowTraffic(f, flowname, flowsize,flowdur, 0,0)
			all_flows.append(flow_created)

		print("flows created")
		print(len(all_flows))

		for fi in all_flows:
			print("Flow is %s, %s, %s, %s" %(fi.num, fi.name,fi.size,fi.dur))

		#create utilization 2D matrix
		perc_eorm_matrix=np.zeros((4,2))
		
		#	avail_util_path=np.zeros([4])
		#next time step gives reward and information on allocat flow, next state is terminal if overflows,
		#  if flow allocated more than capacity, we get a -1, 
		# if still capacity ok +0.1
		avail_util_paths=np.zeros([4])
		avail_util_paths[0]=10
		avail_util_paths[1]=10
		avail_util_paths[2]=8
		avail_util_paths[3]=8
		alloted_eorm_matrix=np.zeros((4,2))
		
		active_flows=[]
		reward=0
		#record action_taken
		rec_action_taken=np.zeros([4])
		rec_flows_chosen=np.zeros([2]) #0 for mouse 1 elephant
		rec_flows_done=np.zeros([2])

		#choose one flow
		flag=0
		


		for i in range(EPISODE_NUMBER): #doesnot do anything now.. depends on flows
			#allocate flow to network
			#get flow
			active_flows=[]
			reward=0
			for fa in all_flows:
				fi=fa
				
				# code do without RL- brute force into it
				#assumption we are allocating the full flow on one path (no splitting)
				
				
				#Choose random action
				#get available bw
				print("available paths")
				print(avail_util_paths) #allocate on first available path
				print("elephant and mice allocated")
				for e in range(4):
					print(alloted_eorm_matrix[e][0])
					print(alloted_eorm_matrix[e][1])
				#path_chosen = random.randint(0,3)
				#take first flow
				
				curr_state=Qtable(avail_util_paths[0], avail_util_paths[1],avail_util_paths[2],avail_util_paths[3], fi.size, 0, 0,
				alloted_eorm_matrix[0][0],alloted_eorm_matrix[0][1],
				alloted_eorm_matrix[1][0], alloted_eorm_matrix[1][1],
				alloted_eorm_matrix[2][0],alloted_eorm_matrix[2][1],
				alloted_eorm_matrix[3][0], alloted_eorm_matrix[3][1])
				
				max_value_found=0
				actn=0
				action_chosen=0

				if random.uniform(0,1)<epsilon:
					actn=random.randint(0,3)  #explore action space
					action_chosen=1
				else: #look for best action to do
					past_action=[]
					for qval in q_table:#self, path0, path1,path2, path3, mOrE, act) #exploit learned values
						if qval.path0==avail_util_paths[0] and qval.path1==avail_util_paths[1] and qval.path2==avail_util_paths[2] and qval.path3==avail_util_paths[3]:
							if qval.m0==alloted_eorm_matrix[0][0] and qval.e0==alloted_eorm_matrix[0][1] and qval.m1==alloted_eorm_matrix[1][0] and qval.e1==alloted_eorm_matrix[1][1]	and qval.m2==alloted_eorm_matrix[2][0] and qval.e2==alloted_eorm_matrix[2][1] and qval.m3==alloted_eorm_matrix[3][0] and qval.e3==alloted_eorm_matrix[3][1]:
								if fi.size<=2 and qval.mOrE<=2: #should be 4 actions in this state
									past_action.append(qval)
								if fi.size>2 and qval.mOrE>2:
									past_action.append(qval)
							#find optimal action in this state
					max_past_reward=0
					action_to_take_from_past=0
					for pa in past_action:
						if max_past_reward<pa.valueQ:
							max_past_reward=pa.valueQ
							action_to_take_from_past=pa.act
							action_chosen=1
							actn=action_to_take_from_past
							curr_state.valueQ=max_past_reward

					# if not found in the qtable
				if action_chosen==0:
					actn=random.randint(0,3) 
					print("chosen random action")


							
					#record state in curr_state

				flag=actn  #get action determined


				if flag==0:
					if avail_util_paths[0]>fi.size:
						avail_util_paths[0] -= fi.size
						flow_to_allocate=FlowTraffic(fi.num, fi.name,fi.size,fi.dur,0,fi.dur+1)
						active_flows.append(flow_to_allocate)
					
						rec_action_taken[0]+=1
						if fi.name=="mouse":
							rec_flows_chosen[0]+=1
							perc_eorm_matrix[0][0]+=1
							alloted_eorm_matrix[0][0]+=1

						else:
							rec_flows_chosen[1]+=1
							perc_eorm_matrix[0][1]+=1
							alloted_eorm_matrix[0][1]+=1
						
						

				if flag==1:
					if avail_util_paths[1]>fi.size:
						avail_util_paths[1]-=fi.size
						flow_to_allocate=FlowTraffic(fi.num, fi.name,fi.size,fi.dur,1,fi.dur+1)
						active_flows.append(flow_to_allocate)
						rec_action_taken[1]+=1
					
						
						if fi.name=="mouse":
							rec_flows_chosen[0]+=1
							perc_eorm_matrix[1][0]+=1
							alloted_eorm_matrix[1][0]+=1
						else:
							rec_flows_chosen[1]+=1
							perc_eorm_matrix[1][1]+=1
							alloted_eorm_matrix[1][1]+=1
						
							#print("allocated 1 %s" %fi.size)

				
				if flag==2:
					if avail_util_paths[2]>fi.size:
						avail_util_paths[2]-=fi.size
						flow_to_allocate=FlowTraffic(fi.num, fi.name,fi.size,fi.dur,2,fi.dur+1)
						active_flows.append(flow_to_allocate)
						rec_action_taken[2]+=1
					
						
						
						if fi.name=="mouse":
							rec_flows_chosen[0]+=1
							perc_eorm_matrix[2][0]+=1
							alloted_eorm_matrix[2][0]+=1

						else:
							rec_flows_chosen[1]+=1
							perc_eorm_matrix[2][1]+=1
							alloted_eorm_matrix[2][1]+=1
						
							#print("allocated 2 %s" %fi.size)


				if flag==3:
					if avail_util_paths[3]>fi.size:
						avail_util_paths[3]-=fi.size
						flow_to_allocate=FlowTraffic(fi.num, fi.name,fi.size,fi.dur,3,fi.dur+1)
						active_flows.append(flow_to_allocate)
						rec_action_taken[3]+=1
					
						

						if fi.name=="mouse":
							rec_flows_chosen[0]+=1
							perc_eorm_matrix[3][0]+=1
							alloted_eorm_matrix[3][0]+=1

						else:
							rec_flows_chosen[1]+=1
							perc_eorm_matrix[3][1]+=1
							alloted_eorm_matrix[3][1]+=1
							#print("allocated 3 %s" %fi.size)

				

				
				#next_st=Qtable(avail_util_paths[0], avail_util_paths[1],avail_util_paths[2],avail_util_paths[3], mflag, flag, 0)
				#claculate reward if any flows finish
				#remove expire flows on path	
				#empty out all active flows allocated
				
				#calculate reward
				for af in active_flows:
					#print("af.counter %s" %af.counter)
					#print("with size %s" %af.size)
					af.counter-=1
					print("af details")
					print(af.size)
					print(af.allocated_to)
					if af.counter<=0:
						lat=0
						if af.allocated_to==0 or af.allocated_to==3:
							lat=3
						if af.allocated_to==1 or af.allocated_to==2:
							lat=1
						
						completion_time=af.dur+lat

						slowness = af.dur/completion_time
						this_reward=slowness
						reward+=this_reward
						#print("flow removed with size %s" %af.size)
						print("Reward %s " %reward)
						print("completed on path %s " %af.allocated_to)
						

						avail_util_paths[af.allocated_to]= avail_util_paths[af.allocated_to]+af.size
					
					
						#remove percentage also
						if af.name=="mouse":
							rec_flows_done[0]+=1
							print("previous %s " %alloted_eorm_matrix[af.allocated_to][0])
							alloted_eorm_matrix[af.allocated_to][0]-=1
						else:
							rec_flows_done[1]+=1
							print("previous %s " %alloted_eorm_matrix[af.allocated_to][1])
							alloted_eorm_matrix[af.allocated_to][1]-=1
						active_flows.remove(af)
				print("out of af loop")

				#calculate new q value
							
				
							
				#end of batch flows
				

				#find next qstate value
				#calculate reward
			
			
				next_state=Qtable(avail_util_paths[0], avail_util_paths[1],avail_util_paths[2],avail_util_paths[3], fi.size, 0, 0,
					alloted_eorm_matrix[0][0],alloted_eorm_matrix[0][1],
					alloted_eorm_matrix[1][0], alloted_eorm_matrix[1][1],
					alloted_eorm_matrix[2][0],alloted_eorm_matrix[2][1],
					alloted_eorm_matrix[3][0], alloted_eorm_matrix[3][1])
				


				#find future state in qtable
				past_action=[]
				for nextqstate in q_table:
					if nextqstate.path0==next_state.path0 and nextqstate.path1==next_state.path1 and nextqstate.path2==next_state.path2 and nextqstate.path3==next_state.path3:
						if nextqstate.m0==next_state.m0 and nextqstate.e0==next_state.e0 and nextqstate.m1==next_state.m1 and nextqstate.e1==next_state.e1 and nextqstate.m2==next_state.m2 and nextqstate.e2==next_state.e2 and nextqstate.m3==next_state.m3 and nextqstate.e3==next_state.e3:
							if nextqstate.mOrE<=2 and next_state.mOrE<=2: #should be 4 actions in this state
								past_action.append(qval)
							if nextqstate.mOrE>2 and next_state.mOrE>2:
								past_action.append(qval)
				max_past_reward=0
				for pa in past_action:
					if max_past_reward<pa.valueQ:
						max_past_reward=pa.valueQ

				curr_state.valueQ=curr_state.valueQ+alpha*(reward+gamma*(max_past_reward-curr_state.valueQ))
				curr_state.act=flag
			
				#find and update Qtable
				found=0
				for qup in q_table:#self, path0, path1,path2, path3, mOrE, act) #exploit learned values
					if qup.path0==curr_state.path0 and qup.path1==curr_state.path1 and qup.path2==curr_state.path2 and qup.path3==curr_state.path3:
						if qup.m0==curr_state.m0 and qup.e0==curr_state.e0 and qup.m1==curr_state.m1 and nextqstate.e1==curr_state.e1 and qup.m2==curr_state.m2 and nextqstate.e2==curr_state.e2 and qup.m3==curr_state.m3 and nextqstate.e3==curr_state.e3:
							if qup.mOrE<=2 and curr_state.mOrE<=2: #should be 4 actions in this state
								if qup.act==flag:
									qup.valueQ=curr_state.valueQ
									found=1
							if qup.mOrE>2 and curr_state.mOrE>2:
								if qup.act==flag:
									qup.valueQ=curr_state.valueQ
									found=1
				if found==0:
					print("adding new q state")
					q_table.append(curr_state)
				
			
		print("****************For episode number %s " %i)
			#print(i)
		for path in range(4):
			print("Percentage flows on path number %s" %path)
			print("mice are %s" %perc_eorm_matrix[path][0])
			print("elephant are %s " %perc_eorm_matrix[path][1])

		print("Next Episode")
		#get reward
			
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
		print("######### All Qtable #######")
		#for qup in q_table:
		#	print(qup.path0)
		#	print(qup.path1)
		#	print(qup.path2)
		#	print(qup.path3)
		#	print(qup.mOrE)
		#	print(qup.act)
		#	print(qup.valueQ)
		#	print(qup.m0)
		#	print(qup.e0)
		#	print(qup.m1)
		#	print(qup.e1)
		#	print(qup.m2)
		#	print(qup.e2)
		#	print(qup.m3)
		#	print(qup.e3)

		print("NextGame")
	#Run agent functions
	return q_table


def test(learned_q_table):

	alpha=0.1
	gamma=0.6
	epsilon=0.1
	
	print("RECEIVED QTABLE")
	for qup in learned_q_table:
		print(qup.path0)
		print(qup.path1)
		print(qup.path2)
		print(qup.path3)
		print(qup.mOrE)
		print(qup.act)
		print(qup.valueQ)
		print(qup.m0)
		print(qup.e0)
		print(qup.m1)
		print(qup.e1)
		print(qup.m2)
		print(qup.e2)
		print(qup.m3)
		print(qup.e3)
	print("size of qtable %s" %len(learned_q_table))

	total_FLOWS=100
	all_flows=[]
		

	#generate elephant or mice flow 100 flows
	for f in range(total_FLOWS):
		EorM= random.randint(0,9)
		#if between 2-9 mouse else elephant
		if(EorM>1):
			flowname="mouse"
			flowsize=random.randint(1,2)
			flowdur=random.randint(1,3)
		else:
			flowname="elephant"
			flowsize=random.randint(3,5)
			flowdur=random.randint(5,8)

		print(flowname,flowsize,flowdur)
		#add active flows to queue
		flow_created=FlowTraffic(f, flowname, flowsize,flowdur, 0,0)
		all_flows.append(flow_created)

	print("flows created")
	print(len(all_flows))

	for fi in all_flows:
		print("Flow is %s, %s, %s, %s" %(fi.num, fi.name,fi.size,fi.dur))

	#create utilization 2D matrix
	perc_eorm_matrix=np.zeros((4,2))
	avail_util_paths=np.zeros([4])
	avail_util_paths[0]=10
	avail_util_paths[1]=10
	avail_util_paths[2]=8
	avail_util_paths[3]=8
	alloted_eorm_matrix=np.zeros((4,2))
	
	active_flows=[]
	reward=0
	#record action_taken
	rec_action_taken=np.zeros([4])
	rec_flows_chosen=np.zeros([2]) #0 for mouse 1 elephant
	rec_flows_done=np.zeros([2])

	#choose one flow
	flag=0
	
	for fa in all_flows:
		fi=fa
				
		# code do without RL- brute force into it
		#assumption we are allocating the full flow on one path (no splitting)
				
				
		#Choose random action#get available bw
		print("available paths")
		print(avail_util_paths) #allocate on first available path
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

		#print("elephant and mice allocated")
		#for e in range(4):
		#	print(alloted_eorm_matrix[e][0])
		#	print(alloted_eorm_matrix[e][1])
		#path_chosen = random.randint(0,3)
		#take first flow
				
		curr_state=Qtable(avail_util_paths[0], avail_util_paths[1],avail_util_paths[2],avail_util_paths[3], fi.size, 0, 0,
		alloted_eorm_matrix[0][0],alloted_eorm_matrix[0][1],
		alloted_eorm_matrix[1][0], alloted_eorm_matrix[1][1],
		alloted_eorm_matrix[2][0],alloted_eorm_matrix[2][1],
		alloted_eorm_matrix[3][0], alloted_eorm_matrix[3][1])
				
		max_value_found=0
		actn=0
		action_chosen=0

		if random.uniform(0,1)<epsilon:
			actn=random.randint(0,3)  #explore action space
			action_chosen=1
		else: #look for best action to do
			past_action=[]
			for qval in learned_q_table:#self, path0, path1,path2, path3, mOrE, act) #exploit learned values
				if qval.path0==avail_util_paths[0] and qval.path1==avail_util_paths[1] and qval.path2==avail_util_paths[2] and qval.path3==avail_util_paths[3]:
					if qval.m0==alloted_eorm_matrix[0][0] and qval.e0==alloted_eorm_matrix[0][1] and qval.m1==alloted_eorm_matrix[1][0] and qval.e1==alloted_eorm_matrix[1][1]	and qval.m2==alloted_eorm_matrix[2][0] and qval.e2==alloted_eorm_matrix[2][1] and qval.m3==alloted_eorm_matrix[3][0] and qval.e3==alloted_eorm_matrix[3][1]:
						if fi.size<=2 and qval.mOrE<=2: #should be 4 actions in this state
							past_action.append(qval)
						if fi.size>2 and qval.mOrE>2:
							past_action.append(qval)
						#find optimal action in this state
			max_past_reward=0
			action_to_take_from_past=0
			for pa in past_action:
				if max_past_reward<pa.valueQ:
					max_past_reward=pa.valueQ
					action_to_take_from_past=pa.act
					action_chosen=1
					actn=action_to_take_from_past
					curr_state.valueQ=max_past_reward
					print("Chosen Action for Qtable %s" %actn)

					# if not found in the qtable
		#if action_chosen==0:
		#	actn=random.randint(0,3) 
		#	print("chosen random action")


							
			#record state in curr_state

		flag=actn  #get action determined


		if flag==0:
			if avail_util_paths[0]>fi.size:
				avail_util_paths[0] -= fi.size
				flow_to_allocate=FlowTraffic(fi.num, fi.name,fi.size,fi.dur,0,fi.dur+1)
				active_flows.append(flow_to_allocate)
					
				rec_action_taken[0]+=1
				if fi.name=="mouse":
					rec_flows_chosen[0]+=1
					perc_eorm_matrix[0][0]+=1
					alloted_eorm_matrix[0][0]+=1

				else:
					rec_flows_chosen[1]+=1
					perc_eorm_matrix[0][1]+=1
					alloted_eorm_matrix[0][1]+=1
						
						

		if flag==1:
			if avail_util_paths[1]>fi.size:
				avail_util_paths[1]-=fi.size
				flow_to_allocate=FlowTraffic(fi.num, fi.name,fi.size,fi.dur,1,fi.dur+1)
				active_flows.append(flow_to_allocate)
				rec_action_taken[1]+=1
					
						
				if fi.name=="mouse":
					rec_flows_chosen[0]+=1
					perc_eorm_matrix[1][0]+=1
					alloted_eorm_matrix[1][0]+=1
				else:
					rec_flows_chosen[1]+=1
					perc_eorm_matrix[1][1]+=1
					alloted_eorm_matrix[1][1]+=1
						
					#print("allocated 1 %s" %fi.size)

				
		if flag==2:
			if avail_util_paths[2]>fi.size:
				avail_util_paths[2]-=fi.size
				flow_to_allocate=FlowTraffic(fi.num, fi.name,fi.size,fi.dur,2,fi.dur+1)
				active_flows.append(flow_to_allocate)
				rec_action_taken[2]+=1
					
						
						
				if fi.name=="mouse":
					rec_flows_chosen[0]+=1
					perc_eorm_matrix[2][0]+=1
					alloted_eorm_matrix[2][0]+=1

				else:
					rec_flows_chosen[1]+=1
					perc_eorm_matrix[2][1]+=1
					alloted_eorm_matrix[2][1]+=1
						
			
		if flag==3:
			if avail_util_paths[3]>fi.size:
				avail_util_paths[3]-=fi.size
				flow_to_allocate=FlowTraffic(fi.num, fi.name,fi.size,fi.dur,3,fi.dur+1)
				active_flows.append(flow_to_allocate)
				rec_action_taken[3]+=1
					
						

				if fi.name=="mouse":
					rec_flows_chosen[0]+=1
					perc_eorm_matrix[3][0]+=1
					alloted_eorm_matrix[3][0]+=1

				else:
					rec_flows_chosen[1]+=1
					perc_eorm_matrix[3][1]+=1
					alloted_eorm_matrix[3][1]+=1
				
				#calculate reward
		for af in active_flows:
			#print("af.counter %s" %af.counter)
			#print("with size %s" %af.size)
			af.counter-=1
			#print("af details")
			#print(af.size)
			print(af.allocated_to)
			if af.counter<=0:
				lat=0
				if af.allocated_to==0 or af.allocated_to==3:
					lat=3
				if af.allocated_to==1 or af.allocated_to==2:
					lat=1
						
				completion_time=af.dur+lat
				slowness = af.dur/completion_time
				this_reward=slowness
				reward+=this_reward
				
				#print("Reward %s " %reward)
				#print("completed on path %s " %af.allocated_to)
						

				avail_util_paths[af.allocated_to]= avail_util_paths[af.allocated_to]+af.size
					
					
				#remove percentage also
				if af.name=="mouse":
					rec_flows_done[0]+=1
					#print("previous %s " %alloted_eorm_matrix[af.allocated_to][0])
					alloted_eorm_matrix[af.allocated_to][0]-=1
				else:
					rec_flows_done[1]+=1
					#print("previous %s " %alloted_eorm_matrix[af.allocated_to][1])
					alloted_eorm_matrix[af.allocated_to][1]-=1
				active_flows.remove(af)


	print("out of af loop and flow loop")
			#print(i)
	for path in range(4):
		print("Percentage flows on path number %s" %path)
		print("mice are %s" %perc_eorm_matrix[path][0])
		print("elephant are %s " %perc_eorm_matrix[path][1])

	print("Next Episode")
	#get reward
	print("total reward at end of 100 flows %s" %reward)		
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

			



def main():
	learned_q_table=[]
	learned_q_table=training(learned_q_table)
	print("****************************IN TESTING PHASE")

	test(learned_q_table)
	print("size of qtable %s" %len(learned_q_table))


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
