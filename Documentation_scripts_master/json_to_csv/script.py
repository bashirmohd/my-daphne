import pandas as pd
from pandas.io.json import json_normalize
import os
import json

## jsonfile is the path to the location of your json file

## read in json file
with open('tmf-2018.json', 'r+') as ff:
    data = json.load(ff)
ff.close()


### make an empty data frame to hold data in csv format
df = pd.DataFrame()

###jsondata is the data youy sent
flows = data['data']['networkEntity']['flow']
len(flows)

for i in range(0, len(flows)): 
    dd = json.loads(flows[i]['traffic']) ### converting the string in traffic to json object
    dd2=pd.DataFrame(dd['points'])
    #dd2['tag'] = ['Traffic{}'.format(i)] * len(dd2.index)
    dd2.columns = dd['columns'] #+ ['tag']
    df = pd.concat([df, dd2], axis=0)

df.to_csv('flows.csv')
