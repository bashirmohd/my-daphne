import numpy as np
from bokeh.layouts import row, column, gridplot
from bokeh.models import ColumnDataSource, Select, Button, HoverTool
from bokeh.plotting import curdoc, figure, show, output_file
from bokeh.driving import count
from bokeh.palettes import Paired10
import itertools

import json
from os.path import dirname, join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import datetime

FILENAME = './17000-in.json'
FILENAME1 = './17700-out.json'
DATA_DIR = join(dirname(__file__), FILENAME)
DATA_DIR1 = join(dirname(__file__), FILENAME1)

def jsonToDF(file):
    initialize = True
    json1_file = open(file)
    json1_str = json1_file.read()
    json1_data = json.loads(json1_str)
    for link in json1_data.keys():
        tsdict = json1_data[link]
        keys = sorted(list(map(int, tsdict.keys())))
        sorted_val = []
        for k in keys:
            sorted_val.append(tsdict[str(k)])
        if initialize:
            datetimes = [datetime.datetime.fromtimestamp(key/ 1e3) for key in keys]
            df = pd.DataFrame({'time': datetimes, link: sorted_val})
            df.set_index('time')
            initialize = False
        else:
            df = pd.concat([df, pd.DataFrame({link: sorted_val})], axis=1)
    return df

#df = jsonToDF(DATA_DIR)
df = jsonToDF(DATA_DIR1)
#df = pd.concat([df, df1], axis=1)
#print(df.head())
#df.plot(figsize=(17,20))
dicts = df.to_dict(orient='records')


headers = list(df.columns.values)
for dict in dicts:
    for k in headers:
        dict[k] = [dict[k]]
newdict = {key: [] for key in headers}
source = ColumnDataSource(newdict)

p = figure(title="ESnet SNMP traffic", plot_height=500, tools="xpan,xwheel_zoom,xbox_zoom,reset,save", x_axis_type="datetime", y_axis_location="right")
p.add_tools(HoverTool(
    tooltips = [
    ("time", "@time{%c}"),
    ("link", "$name"),
    ("bandwith", "$y"),
    ],

    formatters={
        'time'      : 'datetime', # use 'datetime' formatter for 'date' field
        'adj close' : 'printf',   # use 'printf' formatter for 'adj close' field
                                  # use default 'numeral' formatter for other fields
    }
))

# p.x_range.follow = "end"
# p.x_range.follow_interval = 100
# p.x_range.range_padding = 0

links = [link for link in headers if link != 'time']
ts = Select(value="all", options=["all"] + links)
button = Button(label='► Play', width=60)

numlines = len(links)
palette = itertools.cycle(Paired10[0:numlines])

for i in range(numlines):
    p.line(x='time', y=links[i], alpha=0.7, line_width=1, color=next(palette), source=source, name=links[i])
   
@count()
def update(t):
    global dicts, links
    new_data = dicts[t % len(dicts)]
    if ts.value != 'all':
        for k in links:
            if k != ts.value:
                new_data[k] = [0]
    source.stream(new_data, 300)


curdoc().add_root(column(row(button), gridplot([[p]], toolbar_location="left", plot_width=1000), row(ts)))

callback_id = None

def animate():
    global callback_id 
    if button.label == '► Play':
        button.label = '❚❚ Pause'             
        callback_id = curdoc().add_periodic_callback(update, 200)
    else:
        button.label = '► Play'
        curdoc().remove_periodic_callback(callback_id)

button.on_click(animate)

curdoc().title = "SNMP"