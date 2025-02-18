import pandas as pd
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import pacf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import urllib.request
import json
from pathlib import Path

edges = [('AMES', 'STAR'), ('EQX-ASH', 'NETL-PGH'), ('BOIS', 'INL'), ('PNNL', 'PNWG'), ('BOIS', 'PNWG'), 
('EQX-ASH', 'EQX-CHI'), ('LIGO', 'PNWG'), ('BOST', 'PSFC'), ('KANS', 'KCNSC'), ('ATLA', 'ORAU'), ('BOST', 'LNS'), 
('AMST', 'BOST'), ('NERSC', 'SUNN'), ('JLAB', 'WASH'), ('ATLA', 'SRS'), ('HOUS', 'PANTEX'), ('GA', 'SUNN'), ('FNAL', 'STAR'),
 ('LBNL', 'NPS'), ('HOUS', 'KANS'), ('ATLA', 'ETTP'), ('CHIC', 'KANS'), ('ALBQ', 'DENV'), ('JGI', 'SACR'), ('LSVN', 'SUNN'), ('LBNL', 'SUNN'),
 ('ALBQ', 'KCNSC-NM'), ('CHIC', 'STAR'), ('DENV', 'LSVN'), ('DENV', 'NREL'), ('ATLA', 'Y12'), ('DENV', 'KANS'), ('DENV', 'NGA-SW'), ('LSVN', 'NNSS'), 
 ('HOUS', 'NASH'), ('PNWG', 'SACR'), ('CHIC', 'WASH'), ('LLNL', 'SUNN'), ('LOND', 'NEWY'), ('CERN-513', 'CERN-773'), ('ATLA', 'NASH'), ('AMST', 'CERN-513'), 
 ('CERN', 'CERN-513'), ('ANL', 'STAR'), ('PPPL', 'WASH'), ('SLAC', 'SUNN'), ('ATLA', 'ORNL'), ('BOST', 'STAR'), ('ALBQ', 'LANL'), ('NASH', 'WASH'), 
 ('EQX-ASH', 'WASH'), ('AMST', 'LOND'), ('AOFA', 'STAR'), ('AOFA', 'WASH'), ('ELPA', 'HOUS'), ('ELPA', 'SUNN'), ('ALBQ', 'SNLA'), ('SACR', 'SUNN'), 
 ('CERN-773', 'LOND'), ('CHIC', 'NASH'), ('CERN-513', 'WASH'), ('DENV', 'PNWG'), ('EQX-ASH', 'NETL-MGN'), ('AOFA', 'LOND'), ('BNL', 'NEWY'),
  ('CHIC', 'EQX-CHI'), ('ATLA', 'WASH'), ('BOIS', 'DENV'), ('AOFA', 'NEWY'), ('BOST', 'NEWY'), ('ALBQ', 'ELPA'), ('DENV', 'SACR'), ('SACR', 'SNLL')]
url_1 = "https://esnet-netbeam.appspot.com/api/network/esnet/prod/edges/"
url_2 = "/to/"
url_3 = "/timerange?begin=1514793600000&end=1546329599000&rollup=5m"

#unix time to datetime
def ts_to_date(ts):
    str_date = datetime.utcfromtimestamp(ts/1000).strftime('%Y-%m-%d %H:%M:%S')
    return datetime.strptime(str_date,'%Y-%m-%d %H:%M:%S')

#edge is a tuple
#elements of tuple are endpoints of edge
def url_to_csv(edge):

    #url to json
    url = url_1 + edge[0] + url_2 + edge[1] + url_3
    
    #some urls are not there
    try:
        response = urllib.request.urlopen(url)
    except urllib.error.HTTPError as err:
        if err.code == 404:
            print("data for " + edge[0] + "-" + edge[1] + " was not found")
            return
        else: 
            raise

    encoding = response.info().get_content_charset('utf8')
    data = json.loads(response.read().decode(encoding))

    #json to df
    df = pd.DataFrame(data["points"], dtype="float", columns=data["columns"])
    df.set_index('time', inplace=True)
    df.index = df.index.map(ts_to_date)

    #df to csv
    #outdir=Path('one_year_data')
    filename = "one_year_5min_rollups/" + edge[0] + "_" + edge[1] + ".csv"
    df.to_csv(filename)

for edge in edges:
    url_to_csv(edge)
