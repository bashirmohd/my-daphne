import requests
import datetime
import json
import time
import calendar

from google.cloud import firestore


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
def main():

    save5minDict={}

    counter5min=0
    inval=0
    outval=0
    timestamp=0
    
    #end=datetime.datetime.now()+datetime.timedelta(minutes=1)
    while True:
        #endTime=datetime.datetime.now()+datetime.timedelta(minutes=0.5)
        result = run_query(q) 
        js_result=json.dumps(result)
        js_resulta=json.loads(js_result)
        edges=js_resulta["data"]["mapTopology"]["edges"]
        for key in edges:
            if key["name"] == "ALBQ--SNLA":
                #print(key["netbeamTraffic"])
                arr=json.loads(key["netbeamTraffic"])
                print(arr["points"][0][0])
                if timestamp<arr["points"][0][0]:
                    timestamp=arr["points"][0][0]
                    inval+=arr["points"][0][1]
                    outval+=arr["points"][0][2]
                    print("adding")
                #print(arr["points"][0])
        #time.sleep(30)
        counter5min=counter5min+1
        if counter5min>=1:
            break
    print(inval)
    print(outval)
    #add data to firestore

    # Use the application default credentials
    nowtm=datetime.datetime.now()
    nowtimestamp=calendar.timegm(nowtm.utctimetuple())
    print(nowtimestamp)
   
    db = firestore.Client()

    tmpstring='albq--snla--'+str(nowtimestamp)


    doc_ref = db.collection(u'5minrollups').document(tmpstring)
    doc_ref.set({
    u'src':u'ALBQ',
        u'dest':u'SNLA',
        u'timestamp':nowtimestamp,
        u'traffic':inval
    })


     
                
def getData():
    db = firestore.Client()
    users_ref = db.collection(u'5minrollups')
    for doc in users_ref.stream():
        print(u'{} => {}'.format(doc.id, doc.to_dict()))

      
    #result = run_query(q) # Execute the query
    #print(result)


main()

#getData()