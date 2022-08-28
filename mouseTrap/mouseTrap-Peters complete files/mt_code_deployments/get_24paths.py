#calculate three shortest paths and produce 24 values for bar graph
# 
from numpy import array
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
import sys
import re
import time
import json
import numpy as np

import matplotlib.pyplot as plt
import networkx as nx


import logging

from google.cloud import storage
from google.cloud import firestore

#from ml_predictor import simpleLSTM
#global values:
pos = {}
short_five_paths=[]

# UPDATE these to take input from GUI
#src="SUNN"
#dest="CHIC"
#filesize=10

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



def google_map(src,dest):
    #read in map topology
    #build ES.net map

    client = storage.Client()
    globalbucket = client.get_bucket('mousetrap_global_files')
    nodesblob=globalbucket.get_blob('esnet_nodes.json')
    edgesblob=globalbucket.get_blob('esnet_edges.json')

    nodesblobstr=nodesblob.download_as_string()
    edgesblobstr=edgesblob.download_as_string()

    node_dicts = json.loads(nodesblobstr)
    edge_dicts = json.loads(edgesblobstr)
    #print(edge_dicts)

    G = build_graph(node_dicts['data']['mapTopology']['nodes'], edge_dicts['data']['mapTopology']['edges'])
    #pos=fill_pos(node_dicts)
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


    short_five_paths.append(paths_list[0])
    short_five_paths.append(paths_list[1])
    short_five_paths.append(paths_list[2])
    short_five_paths.append(paths_list[3])
    short_five_paths.append(paths_list[4])
   
    
    return short_five_paths


def build_edge_map():
    client = storage.Client()
    globalbucket = client.get_bucket('mousetrap_global_files')
    edgesblob=globalbucket.get_blob('esnet_edges.json')

    edgesblobstr=edgesblob.download_as_string()

    edge_dicts = json.loads(edgesblobstr)
    #print(edge_dicts)
    edges=edge_dicts['data']['mapTopology']['edges']
    
    return edges

class edge_predictions:
    def __init__(self, name, src,dest):
        self.name=name
        self.src=src
        self.dest=dest
        self.timestamp=[]
        self.mean=[]
        self.conf=[]

    def add_timestamp(self,timestamp):
        self.timestamp.append(timestamp)

    def add_mean(self,mean):
        self.mean.append(mean)

    def add_conf(self,conf):
        self.conf.append(conf)

    
def get_latest_data(src,dest):

    edges=build_edge_map()

    db = firestore.Client()
    #query on Date with descending (limit 1)

    users_ref = db.collection(u'latest-predictions')
    #query=users_ref.where(u'src', u'==',src).where(u'dest',u'==',dest).stream()
    #query_last=query#.order_by(u'timestamp')#, direction=firestore.Query.DESCENDING).limit(1)
    #results=query.stream()
    #print(results)
    
    
    latest_time=0
    edgeCalendar=[]
    for edge in edges:
        if receivedDict['src']==edge['ends'][0]['name'] and receivedDict['dest']==edge['ends'][1]['name']:
            #print("Found EDGE")
            strname=edge['ends'][0]['name']+'--'+edge['ends'][1]['name']
            #print(strname)
            #print(u'{} => {}'.format(doc.id, doc.to_dict()))
            flag=0
            for j in edgeCalendar:
                if j.src==edge['ends'][0]['name'] and j.dest==edge['ends'][1]['name']:
                    j.timestamps.append(receivedDict['timestamp'])
                    j.values.append(receivedDict['traffic'])
                    flag=1
            if flag==0:
                edgeCalendar.append(edge_totals(strname,edge['ends'][0]['name'],edge['ends'][1]['name'],receivedDict['timestamp'],receivedDict['traffic']))
                
    return edgeCalendar


def get_latest_blob():

    storage_client=storage.Client()
    pred_bucket='latest24predictions-esnet'

    latest_time=0
    latest_blob='temp'

    blobs=storage_client.list_blobs(pred_bucket)
    for blob in blobs:
        print(blob.name)
        ts=blob.updated
        print(blob.updated)
        print(ts.timestamp())
        if ts.timestamp()>latest_time:
            latest_time=ts.timestamp()
            latest_blob=blob.name
    
    print("latest is")
    print(latest_blob)

    bucket=storage_client.get_bucket(pred_bucket)
    blob=bucket.get_blob(latest_blob)
    json_file= str(blob.download_as_string(),'utf-8')
    #print(json_file)
    #json_file is string
    json_file=json_file.replace("'",'"')
    print("convert to json")
    json_file=json.loads(json_file)#,ensure_ascii=False).encode('utf8'))
    #json_dataObj=json.loads(json_file)
    #print(json_dataObj)

    print("_________")
    #for key in json_file:
    #   value= json_file[key]
    #print(json_file["data"][0])
        #print("key and value ({}) = ({})".format(key,value))
    return json_file


def main(src, dest, filesize):
    #run ML model
    #read 24 link values

    start_time=time.time()

    #get latest predictions
    pred_24=get_latest_blob()
    #print(pred_24)
    #edgeCalendar1=get_latest_data(src,dest)
    
    #draw graph and hightlight three paths
    #get SRC and Dest
    print("Source: ")
    print(src)
    print("Destination: ")
    print(dest)
    print("File size (GB):")
    print(filesize)
    
    #returns three short paths:
    short_five_paths=google_map(src,dest)
    print("SHORTEST PATHS aRE:")
    for sf in short_five_paths:
        print(sf)
   
    #loop through and claculate the json for 24 values
    
    calculatedvalues=np.zeros((5,24))

    
    result24Json={}
    data=[]
    newdata=[]
    
    tm="value"
    div100G=858993459200 #Bits
   
    counthr=0
    fulldata=[]
    for hr in range(24):
        print("HOUR")
        hrpreddata={}

        print(hr)
        savedroad=[]
        savedtime=0
        savedcolorsarray=[]
        saved_highest_perc=0
        #calculate shortest path in every hour
        sumhr=10000000000000000000000000000000000000000000000000000

        for road in short_five_paths:
            print("*************************one Road ***************************")
            print(road)
            totalhr=0
            colorsarray=[]
            highest_perc=0

            for lanei, lanej in zip(road,road[1:]):
                for p in pred_24['data']:
                    if p['src']==lanei and p['dest']==lanej:
                        print(lanei)
                        print(lanej)
                        print("pvalue:")
                        print(p['values'][hr])
                        perc=(p['values'][hr]/div100G)
                        totalhr+=p['values'][hr]
                        print(perc)
                        colorsarray.append(perc)
                        if perc>highest_perc:
                            highest_perc=perc
                            
            print("total for this path: ")
            print(totalhr)
            print(colorsarray)
            print(highest_perc)
            #saving best path for this hour
            if totalhr<sumhr:
                print("shortest")
                print(totalhr)
                print(sumhr)
                sumhr=totalhr
                savedroad=road[:]
                savedtime=totalhr
                savedcolorsarray=colorsarray[:]
                saved_highest_perc=highest_perc
        #best path found
        print("Shortest path for this hour is")   
        print(savedroad)
        print(savedtime) 
        print(savedcolorsarray)
        print(saved_highest_perc)

        tmp_dict={}
        data=[]
        
        tmp_dict['path']=savedroad
        tmp_dict['predictedtotal']=savedtime
        tmp_dict['weights']=savedcolorsarray #[0,0,0,0]#emparray
        tmp_dict['bottleneck']=saved_highest_perc

        data.append(tmp_dict)

        hrpreddata["hour"]=hr
        hrpreddata["data"]=data
        fulldata.append(hrpreddata)
    
    result24Json={}
    result24Json["predictions"]=fulldata
    end_time=time.time()
    total_time=end_time-start_time
    print("total_time: %s seconds" %total_time)
    return result24Json
   






js=main("LBNL","BNL",10)



