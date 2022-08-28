import requests
import datetime
import json
import time
import calendar

from google.cloud import firestore
from google.cloud import storage


def run_query(query): # A simple function to use requests.post to make the API call. Note the json= section.
    request = requests.get("https://my.es.net/graphql?", params={'query':q})
    print(request)
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, query))

        
# The GraphQL query (with a few aditional bits included) itself defined as a multi-line string.       
q = """
{
    mapTopology(name: "routed_toplevel") {
        edges {
            name
            netbeamTraffic
        }
    }
}
"""

class edge_totals:
    def __init__(self, name, src,dest,timestamp, invalue,outvalue):
        self.name=name
        self.src=src
        self.dest=dest
        self.timestamp=timestamp
        self.invalue=invalue
        self.outvalue=outvalue


def build_edge_map():
    client = storage.Client()
    globalbucket = client.get_bucket('mousetrap_global_files')
    edgesblob=globalbucket.get_blob('esnet_edges.json')

    edgesblobstr=edgesblob.download_as_string()

    edge_dicts = json.loads(edgesblobstr)
    #print(edge_dicts)
    edges=edge_dicts['data']['mapTopology']['edges']
    
    return edges
    
        


def main():

    save5minDict={}

    counter5min=0
    inval=0
    outval=0
    timestamp=0

    #build an edge map
    edges=build_edge_map()
    #print(len(edges))

    edge_n=len(edges)

    edge_list=[]

    for edge in edges:
        edge_list.append(edge_totals(edge['name'],edge['ends'][0]['name'],edge['ends'][1]['name'],0,0,0))
        #print(edge['name'])

    #for n in range(edge_n):
        #print(edge_list[n].name)
    
    #end=datetime.datetime.now()+datetime.timedelta(minutes=1)
    hrcounter=0

    while True:
        #endTime=datetime.datetime.now()+datetime.timedelta(minutes=0.5)
        result = run_query(q) 
        js_result=json.dumps(result)
        js_resulta=json.loads(js_result)
        read_edges=js_resulta["data"]["mapTopology"]["edges"]

        #print("query rture")
        #print(read_edges)

        for key in read_edges:
            for el in edge_list:
                if key["name"] == el.name:
                #print(key["netbeamTraffic"])
                    arr=json.loads(key["netbeamTraffic"])
                    #print(arr["points"][0][0])
                    if el.timestamp<arr["points"][0][0]:
                        el.timestamp=arr["points"][0][0]
                        el.invalue+=arr["points"][0][1]
                        el.outvalue+=arr["points"][0][2]
                        #print("adding")
                        #print(el.name)
                    
        time.sleep(30)
        #print("saved all")
        hrcounter=hrcounter+1
        #print("Printing..........")

        if(hrcounter>=10): #120 #10
            #save to file
            nowtm=datetime.datetime.now()
            nowtimestamp=calendar.timegm(nowtm.utctimetuple())
            print("saved")
            print(nowtimestamp)
            db = firestore.Client()


            for eln in edge_list:
                #print(eln.name)
                #print(eln.src)
                #print(eln.dest)
                #print(eln.invalue)
                #print(eln.outvalue)

                tmpstring=eln.name+'--'+str(nowtimestamp)
                doc_ref = db.collection(u'testrollups').document(tmpstring)
                doc_ref.set({
                    u'src':eln.src,
                    u'dest':eln.dest,
                    u'timestamp':nowtimestamp,
                    u'traffic':eln.invalue
                })
                #flip dect-src
                new_tmpstring=eln.dest+'--'+eln.src+'--'+str(nowtimestamp)
                doc_ref = db.collection(u'testrollups').document(new_tmpstring)
                doc_ref.set({
                    u'src':eln.dest,
                    u'dest':eln.src,
                    u'timestamp':nowtimestamp,
                    u'traffic':eln.outvalue
                })


            
            hrcounter=0

    
    
    
    
   # print(inval)
   # print(outval)
    #add data to firestore

    # Use the application default credentials
   



main()