import json
import os
import pandas as pd
import matplotlib
from datetime import datetime
import numpy as np

#Name (CHP7), start, end, duration (calculate),eros, three points from Richards files)

name = 'sc18.json'
f = open(name).read()
f = f.replace('\n', '')
f = f.replace('\t', '')
f = f.replace(' ', '')
d = eval(f)
df = pd.DataFrame.from_dict(d, orient='index')
df = df[['start', 'end', 'eros']]
df['duration'] = df['end'].astype(int) - df['start'].astype(int)
df['points'] = np.empty((len(df), 0)).tolist()
df.index.name = "name"
# json_data = json.loads(f)
possible_files = ['data/' + name + '.json' for name in df.index]
for name in df.index:
    path = 'data/' + name + '.json'
    if(os.path.isfile(path)):
        f = open(path)
        json_data = json.loads(f.read())
        points = json_data['points']
        df.at[name, 'points'] = points

df.to_csv('oscars_data.csv')


