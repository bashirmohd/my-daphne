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
df.head()
df.tail()
df = df.dropna()
df.shape
df.isnull().sum()

df.dtypes

df.plot(figsize=(12,8))
plt.xlabel('Time (Hours)')
plt.ylabel('Bandwidth Utilization(Gbps)')
plt.title('Bandwidth Utitilization Over Time for 3-Months')

df['Days'] = pd.to_datetime(df['Days'],infer_datetime_format=True)
df = df.set_index('Days')
df.head()

#Test for Stationarity
def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")


df.columns
# ADT Test for each link
adf_test(df['aofa_lon_In_speed'])
adf_test(df['aofa_lon_out_speed'])
adf_test(df['cern_wash_in_speed'])
adf_test(df['cern_wash_out_speed'])
adf_test(df['lond_newy_in_speed']) 
adf_test(df['lond_newy_out_speed']) 
adf_test(df['amst_bost_in_speed']) 
adf_test(df['amst_bost_out_speed']) 

from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
decomposition = sm.tsa.seasonal_decompose(df['aofa_lon_In_speed'], model='additive')
fig = decomposition.plot()
plt.show()

from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
decomposition = sm.tsa.seasonal_decompose(df['aofa_lon_out_speed'], model='additive')
fig = decomposition.plot()
plt.show()

from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
decomposition = sm.tsa.seasonal_decompose(df['cern_wash_in_speed'], model='additive')
fig = decomposition.plot()
plt.show()

from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
decomposition = sm.tsa.seasonal_decompose(df['cern_wash_out_speed'], model='additive')
fig = decomposition.plot()
plt.show()

from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
decomposition = sm.tsa.seasonal_decompose(df['lond_newy_in_speed'], model='additive')
fig = decomposition.plot()
plt.show()

from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
decomposition = sm.tsa.seasonal_decompose(df['lond_newy_out_speed'], model='additive')
fig = decomposition.plot()
plt.show()

from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
decomposition = sm.tsa.seasonal_decompose(df['amst_bost_in_speed'], model='additive')
fig = decomposition.plot()
plt.show()

from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
decomposition = sm.tsa.seasonal_decompose(df['amst_bost_out_speed'], model='additive')
fig = decomposition.plot()
plt.show()

df.shape

average_daily_speed = df.resample('D').mean()
average_daily_speed.dtypes

average_daily_speed.plot()
plt.title('Average daily Speed (Gbps)')
plt.show()

average_daily_speed.head()

df_av = average_daily_speed

df_av.head()

df_av.shape

# NUm of Observations
nobs = 24

train = df_av[:-nobs] #Start=  begining of df--> -24 from the end

test = df_av[-nobs:]# start -24 from the end of the DF ---> go to the end of DF

print(train.shape)
print(test.shape)

# to determin the order of P i.e lag order, and k=8 becoz we have 8 columns
for i in [1,2,3,4,5,6]:
    model = VAR(train)
    results = model.fit(i)
    print('Order =', i)
    print('AIC: ', results.aic)
    print('BIC: ', results.bic)
    print()

results = model.fit(6)  #p=6

results.summary()


results.plot_acorr()
plt.show()

model.select_order(6)

results = model.fit(maxlags=6, ic='aic')

lag_order = results.k_ar

results.plot_forecast(24)
plt.show()





