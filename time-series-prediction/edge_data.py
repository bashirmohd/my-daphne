import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime
import re

# a simple function to use requests to make the API call
# returns json content from query
def run_query(query):
    request = requests.get('https://my.es.net/graphql', json={'query': query})
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, query))

#query a specified edge for a specified time period
def query_edge(id, beginTime, endTime):
    query = '''
    {
      mapTopologyEdge(id: "%d") {
            name
        traffic(beginTime: "%s", endTime: "%s") {
          columns
          name
          points
          utc
          labels
          interface
          device
          sap
          tile
        }
      }
    }
    ''' % (id, beginTime, endTime)
    data = run_query(query)
    return data

#return a dataframe
#columns: time, traffic from site1->site2, traffic from site2->site1
def clean_edge_data(json_data):
    #get "traffic" label from json data
    traffic = str(json_data['data']['mapTopologyEdge']['traffic'])

    #get index of "points" label
    index = traffic.find("points")

    #get "points" data
    in_out = traffic[index+len("points: "):].split("],")
    points = [str(item).strip().replace("[", "").replace("]", "").replace(",", "").split() for item in in_out]
    points = points[0:len(points)-1]

    #get labels of points data (ex. CHIC--STAR, STAR--CHIC)
    labels = traffic[traffic.find("labels")+len("labels: "):].split("],")[0].replace("[", 
                                            "").replace("]", "").replace(",", "").replace('"', "").strip().split()
    #Add "Time" to list for header in DataFrame
    labels = ["Time"] + labels
    
    #Create DataFrame
    df = pd.DataFrame(points[:len(points)-2], columns=labels)
    df = df.astype('float')
    df['Time'] = df['Time'].apply(lambda x: datetime.fromtimestamp(x/1000.))
    
    return df

#return df of traffic in a single direction
#specify weekend, day, hour, minute
def edge_features(df, pathway):
    #Get only one pathway (CHIC--STAR)
    single_pathway = df[['Time', pathway]]

    #Convert times to datetime objects
    times = single_pathway['Time']

    #Is date weekend?
    weekends = times.apply(lambda x: x.weekday() >= 5)*1
    single_pathway['Weekend'] = weekends

    #Add day of the week
    days = times.apply(lambda x: x.weekday())
    single_pathway['Day'] = days

    #Add hours, minutes
    hours = times.apply(lambda x: x.hour)
    minutes = times.apply(lambda x: x.minute + x.second/60.)
    single_pathway['Hour'] = hours
    single_pathway['Minute'] = minutes

    return single_pathway

edge_1894 = query_edge(1894, "2018-05-20T18:22:53.253Z", "2018-05-21T19:22:53.253Z")
points_1894 = clean_edge_data(edge_1894)
points_1894_denvpnwg = edge_features(points_1894,"'DENV--PNWG'")