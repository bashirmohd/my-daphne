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
from RLtechniques.dqnclasses import GameState

import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim




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

def init_weights(m):
	if type(m) == nn.Conv2d or type(m) == nn.Linear:
		torch.nn.init.uniform(m.weight, -0.01, 0.01)
		m.bias.data.fill_(0.01)

def image_to_tensor(na):
	#x=torch.Tensor(a,b,c,d,m,ac)
	#image_tensor=image.transpose(2,0,1)
	#image_tensor=image_tensor.astype(np.float32)
	image_tensor=torch.from_numpy(na)
	return image_tensor


def training():
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
	TRAININGITERATION=5
	EPISODE_NUMBER=1
	total_FLOWS=10
	#1 gam finishes when 100 flows are sent to destination
	# #initialize 2D matrix
	# #Demand_matrix=Current utilization matrix + incoming flow
	
	alpha=0.1
	gamma=0.6
	epsilon=0.1
	
	all_epochs=[]
	

	model = NeuralNetwork()
	model.apply(init_weights)
	start = time.time()

	optimizer = optim.Adam(model.parameters(), lr=1e-6)
	# initialize mean squared error loss
	criterion = nn.MSELoss()


	#for training the game agent to play the game multiple times
	for j in range(TRAININGITERATION):
		#one game iteration start to terminal state

		iter_training=0


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
				flowdur=random.randint(10,15)

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

		active_flows=[]
		#record action_taken
		rec_action_taken=np.zeros([4])
		rec_flows_chosen=np.zeros([2]) #0 for mouse 1 elephant
		rec_flows_done=np.zeros([2])

		#choose one flow
		flag=0

		#game initialized above
		#initialize replay memory
		replay_memory=[]


		for i in range(EPISODE_NUMBER): #doesnot do anything now.. depends on flows
			#allocate flow to network
			#get flow
			while len(all_flows)>0:
				#get current utilization matrix
				fi=all_flows[0]
				

				if(len(all_flows)==100):
					print("100 step")
					print(avail_util_paths)
				if(len(all_flows)==90):
					print("90 step")
					print(avail_util_paths)
				if(len(all_flows)==80):
					print("80 step")
					print(avail_util_paths)
				if(len(all_flows)==70):
					print("70 step")
					print(avail_util_paths)
				if(len(all_flows)==60):
					print("60 step")
					print(avail_util_paths)
				if(len(all_flows)==50):
					print("50 step")
					print(avail_util_paths)
				if(len(all_flows)==40):
					print("40 step")
					print(avail_util_paths)
				if(len(all_flows)==30):
					print("30 step")
					print(avail_util_paths)
				if(len(all_flows)==20):
					print("20 step")
					print(avail_util_paths)
				if(len(all_flows)==10):
					print("10 step")
					print(avail_util_paths)
				if(len(all_flows)==1):
					print("1 step")
					print(avail_util_paths)
				# code do without RL- brute force into it
				#assumption we are allocating the full flow on one path (no splitting)
				
				
				#Choose random action
				#get available bw
				print("available paths")
				print(avail_util_paths) #allocate on first available path
				#path_chosen = random.randint(0,3)
				#take first flow
				#get current state of game
				if fi.name=="mouse":
					me=0
				else:
					me=1

				curr_state=GameState(avail_util_paths[0], avail_util_paths[1],avail_util_paths[2],avail_util_paths[3], me,0)
				epsilon = model.initial_epsilon

				epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)

				narr=np.array([avail_util_paths[0], avail_util_paths[1],avail_util_paths[2],avail_util_paths[3], me],dtype=np.float32)
				
				data = torch.from_numpy(narr)
				# imagedata=image_to_tensor(narr)
				print(data.shape)
				batch = data.unsqueeze(0)
				print(batch.shape)
				# cstate=torch.cat((imagedata,imagedata,imagedata,imagedata)).unsqueeze(0)

				# print(cstate.shape)

				# cstate=cstate.view(-1)

				# print("flatten") #tensor is already 24 size
				# print(cstate)
				# cstate=cstate.expand(30,24,1,1)
				# print(cstate)

				# output = model(cstate)[0]
				output = model(batch)

				print("output", output.shape)
				print(output)

				#initial action
				action=torch.zeros([model.number_of_actions], dtype=torch.float32)

				#epsilon greedy exploration
				random_action = random.random() <= epsilon
				if random_action:
					print("Performed random action!")
				
				action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
				
				if random_action
				else torch.argmax(output)][0]


				
				action[action_index] = 1

				print("actionindex")
				print(action[action_index])

				max_value_found=0
				actn=0

				#record state in curr_state

				flag=actn  #get action determined

				if flag==0:
					if avail_util_paths[0]>fi.size:
						avail_util_paths[0] -= fi.size
						flow_to_allocate=FlowTraffic(fi.num, fi.name,fi.size,fi.dur,0,fi.dur+1)
						active_flows.append(flow_to_allocate)
					
						all_flows.remove(fi)
						perc=fi.size/10

						rec_action_taken[0]+=1
						if fi.name=="mouse":
							rec_flows_chosen[0]+=1
							perc_eorm_matrix[0][0]+=perc

						else:
							rec_flows_chosen[1]+=1
							perc_eorm_matrix[0][1]+=perc
						
						print("allocated 0 %s" %fi.size)

				if flag==1:
					if avail_util_paths[1]>fi.size:
						avail_util_paths[1]-=fi.size
						flow_to_allocate=FlowTraffic(fi.num, fi.name,fi.size,fi.dur,1,fi.dur+1)
						active_flows.append(flow_to_allocate)
						rec_action_taken[1]+=1
					
						all_flows.remove(fi)
						perc=fi.size/10
						if fi.name=="mouse":
							rec_flows_chosen[0]+=1
							perc_eorm_matrix[1][0]+=perc
						else:
							rec_flows_chosen[1]+=1
							perc_eorm_matrix[1][1]+=perc
						
						print("allocated 1 %s" %fi.size)

				
				if flag==2:
					if avail_util_paths[2]>fi.size:
						avail_util_paths[2]-=fi.size
						flow_to_allocate=FlowTraffic(fi.num, fi.name,fi.size,fi.dur,2,fi.dur+1)
						active_flows.append(flow_to_allocate)
						rec_action_taken[2]+=1
					
						all_flows.remove(fi)
						perc=fi.size/8

						if fi.name=="mouse":
							rec_flows_chosen[0]+=1
							perc_eorm_matrix[2][0]+=perc

						else:
							rec_flows_chosen[1]+=1
							perc_eorm_matrix[2][1]+=perc
						
						print("allocated 2 %s" %fi.size)


				if flag==3:
					if avail_util_paths[3]>fi.size:
						avail_util_paths[3]-=fi.size
						flow_to_allocate=FlowTraffic(fi.num, fi.name,fi.size,fi.dur,3,fi.dur+1)
						active_flows.append(flow_to_allocate)
						rec_action_taken[3]+=1
					
						all_flows.remove(fi)
						perc=fi.size/8

						if fi.name=="mouse":
							rec_flows_chosen[0]+=1
							perc_eorm_matrix[3][0]+=perc

						else:
							rec_flows_chosen[1]+=1
							perc_eorm_matrix[3][1]+=perc
						print("allocated 3 %s" %fi.size)

				mflag=0
				if fi.name=="mouse":
					mflag=0
				if fi.name=="elephant":
					mflag=1

				curr_state.mOrE=mflag
				curr_state.act=flag
				next_st=Qtable(avail_util_paths[0], avail_util_paths[1],avail_util_paths[2],avail_util_paths[3], mflag, flag, 0)

				#claculate reward if any flows finish
				#remove expire flows on path	
				for af in active_flows:
					#print("af.counter %s" %af.counter)
					#print("with size %s" %af.size)
					af.counter-=1

					if af.counter<=0:
						lat=0
						if af.allocated_to==0 or af.allocated_to==3:
							lat=3
						if af.allocated_to==1 or af.allocated_to==2:
							lat=1
						
						completion_time=af.dur+lat

						slowness = af.dur/completion_time
						this_reward=slowness+1
						reward+=this_reward
						print("flow removed with size %s" %af.size)
						print("Reward %s " %reward)
						

						avail_util_paths[af.allocated_to]= avail_util_paths[af.allocated_to]+af.size
						active_flows.remove(af)
						new_perc=0
						#remove percentage also
						if af.allocated_to==0 or af.allocated_to==1:
							new_perc= af.size/10
						else:
							new_perc=af.size/8

						if af.name=="mouse":
							rec_flows_done[0]+=1
							perc_eorm_matrix[af.allocated_to][0]-=new_perc
						else:
							rec_flows_done[1]+=1
							perc_eorm_matrix[af.allocated_to][1]-=new_perc
				
							
				#calculate penalty
				penal_change=0
				for row in range(4):
					if perc_eorm_matrix[row][1]!=0:
						if perc_eorm_matrix[row][0]>perc_eorm_matrix[row][1]:
							#too many mouse than elephant , penalize
							penalty+=1
							penal_change=1
					#print("congestion matrix %s " %perc_eorm_matrix[row][0])
					#print("congestion matrix2 %s " %perc_eorm_matrix[row][1])

				print("Penalty %s" %penalty)

				# save transition to replay memory
				replay_memory.append((curr_state, flag, reward, next_st))

				# if replay memory is full, remove the oldest transition
				if len(replay_memory) > model.replay_memory_size:
					replay_memory.pop(0)

				# epsilon annealing
				epsilon = epsilon_decrements[iteration]

				# sample random minibatch
				minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

				# unpack minibatch
				state_batch = torch.cat(tuple(d[0] for d in minibatch))
				action_batch = torch.cat(tuple(d[1] for d in minibatch))
				reward_batch = torch.cat(tuple(d[2] for d in minibatch))
				state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

				# get output for the next state
				output_1_batch = model(state_1_batch)
				# set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
				y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
							else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
							for i in range(len(minibatch))))

				# extract Q-value
				q_value = torch.sum(model(state_batch) * action_batch, dim=1)

				# PyTorch accumulates gradients by default, so they need to be reset in each pass
				optimizer.zero_grad()

				# returns a new Tensor, detached from the current graph, the result will never require gradient
				y_batch = y_batch.detach()

				# calculate loss
				loss = criterion(q_value, y_batch)

				# do backward pass
				loss.backward()
				optimizer.step()

				# set state to be state_1
				state = state_1
				iteration += 1

				#find next qstate value
				next_q_value=0
				for nexqval in q_table:#self, path0, path1,path2, path3, mOrE, act) #exploit learned values
					if nexqval.path0==next_st.path0 and  nexqval.path1==next_st.path1 and  nexqval.path2==next_st.path2 and nexqval.path3==next_st.path3:
						if nexqval.mOrE==nexqval.mOrE:
							next_q_value=nexqval.valueQ

				#update value for Q
				new_val=(1-alpha) * curr_state.valueQ+alpha * (reward + gamma * next_q_value)

				#adding for penalty# comment out to REMOVE
				if penal_change==1:# if penalty occured do not add any rewards to RL
					new_val=(1-alpha) * curr_state.valueQ #+ alpha * (reward + gamma * next_q_value) - alpha * (1 + gamma * next_q_value)
				
				print("new val %s" %new_val)
				curr_state.valueQ=new_val

							
				

				print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
					action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
					np.max(output.cpu().detach().numpy()))
			
				print("end while: size of flows %s " %len(all_flows))

			#print(i)
			print("Next Episode")
			#get reward
			
			print("reward %s" %reward)
			print("penalty %s" %penalty)
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

def test():
	print("test")

def main():
	training()
	test()



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
