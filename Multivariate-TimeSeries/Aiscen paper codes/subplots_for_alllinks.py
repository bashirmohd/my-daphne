import pandas as pd
import numpy as np
import matplotlib
#%matplotlib inline
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
import itertools
warnings.filterwarnings("ignore")
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] =12
matplotlib.rcParams['text.color'] = 'k'

# Load specific forecasting tools
from statsmodels.tsa.api import VAR, DynamicVAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse

path_to_data = 'https://raw.githubusercontent.com/bashirmohd/DataSceince/master/Time-series-prediction_production/all_links.csv'

df = pd.read_csv(path_to_data)

df['Days'] = pd.to_datetime(df['Days'],infer_datetime_format=True)

df = df.set_index('Days')

df.columns

fig,ax = plt.subplots(8,1)
for i,column in enumerate([col for col in df.columns]):
    df[column].plot(ax=ax[i])
    ax[i].set_title(column)