import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing
# from save_csv_and_plot import *

def normalize(df):
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    for col in df.columns:
        null_index = df[col].isnull()
        df.loc[~null_index, [col]] = scaler.fit_transform(df.loc[~null_index, [col]])
    return df

def plot_curtail_and_price(cur_col, price_col, df):
    plt.subplot(1, 2, 1)
    plt.plot(df.index, df[cur_col], 'b-')
    plt.title(cur_col)
    plt.xlabel('Date')
    plt.ylabel('Curtailment')
    plt.ylim((0,1))
    plt.xticks(rotation=70)

    plt.subplot(1, 2, 2)
    plt.plot(df.index, df[price_col], 'r-')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.ylim((0,1))
    plt.xticks(rotation=70)

    plt.show()

df = pd.read_csv('curtailment_and_prices.csv', index_col=0)
df = df.apply(pd.to_numeric)
df.index =df.index.map(lambda t: t[:13])
#if we want to normalize
df = normalize(df)

#--SAMPLE PLOT--
# plot_curtail_and_price("Unit 7&8", "Unit7_8_price", df[43800:43830])

#--PLOT PER TRANSMISSION--
lakeville = ["Calistoga", "Sonoma", "Unit 18", "Unit 13", "Unit 20"]
fulton = ["Unit 12", "Unit 14", "Unit 16", "Bear Canyon", "Unit 17"]
eagle_rock = ["Aidlin", "Unit 5&6", "Unit 7&8", "Unit 11"]

cur = df[32040:len(df)-8760]
for col in lakeville:
    plt.plot(cur[col], label=col)
plt.legend()
plt.show()

for col in fulton:
    plt.plot(cur[col], label=col)
plt.legend()
plt.show()

for col in eagle_rock:
    plt.plot(cur[col], label=col)
plt.legend()
plt.show()
