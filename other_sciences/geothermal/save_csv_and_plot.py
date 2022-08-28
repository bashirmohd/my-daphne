import pandas as pd
import numpy as np
from os import listdir
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing

#pricing data has repeated dates due to daylight savings
#average these values
#note: curtailment data does not have repeated dates
def average_duplicates(df):
    if(df.index.is_unique==False):
        df = df.groupby(df.index).mean()
    return df

def normalize(data):
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    data = [[d] for d in data]
    scaler.fit(data)
    data = scaler.transform(data)
    data = [j for i in data for j in i]
    return data

def price_df(filename):
    df = pd.read_csv(filename, index_col=0)
    df['Local Datetime (Hour Beginning)'] = pd.to_datetime(df['Local Datetime (Hour Beginning)'])
    df["Price $/MWh"] = pd.to_numeric(df['Price $/MWh'])
    df.set_index('Local Datetime (Hour Beginning)', inplace=True)
    price_site = filename[7:len(filename)-4]
    df.rename(columns={'Price $/MWh': price_site}, inplace=True)
    df = df[price_site]
    return price_site, pd.DataFrame(average_duplicates(df))

#handle cases where there are 2 price nodes corresponding to 1 curtailment node
def join_price_nodes(df1, df2, newname):
    df3 = pd.concat([df1,df2], axis=1)
    df3[newname] = df3.mean(axis=1)
    df3 = pd.DataFrame(df3[newname])
    return df3

def plot_curtail_and_price(cur_col, price_col, df):
    plt.subplot(1, 2, 1)
    plt.plot(df.index, df[cur_col], 'b-')
    plt.title(cur_col)
    plt.xlabel('Date')
    plt.ylabel('Curtailment')

    plt.subplot(1, 2, 2)
    plt.plot(df.index, df[price_col], 'r-')
    plt.xlabel('Date')
    plt.ylabel('Price')

    plt.show()

price_files = ["prices/" + f for f in listdir("prices")]
#.DS_Store at position 0
price_files = price_files[1:]
price_dfs = {}

for filename in price_files:
    price_site, df = price_df(filename)
    price_dfs[price_site] = df

sf1 = price_dfs.pop('GeysersPrices-SANTAFE_7_B1')
sf2 = price_dfs.pop("GeysersPrices-SANTAFE_7_B11")

u561 = price_dfs.pop('GeysersPrices-GEYSR5-6_7_B2')
u562 = price_dfs.pop("GeysersPrices-GEYSR5-6_7_B3")

u781 = price_dfs.pop('GeysersPrices-GEYSER78_7_B1')
u782 = price_dfs.pop("GeysersPrices-GEYSER78_7_B3")

sf_avg = join_price_nodes(sf1, sf2, "Calistoga_price")
u56_avg = join_price_nodes(u561, u562, "Unit5_6_price")
u78_avg = join_price_nodes(u781, u782, "Unit7_8_price")

mapping = {
    "GeysersPrices-SMUDGEO1_7_B1": "Sonoma_price",
    "GeysersPrices-GEYSER20_7_B1": "Unit20_price",
    "GeysersPrices-GEYSER18_7_B1": "Unit18_price",
    "GeysersPrices-GEYSER17_7_B1": "Unit17_price",
    "GeysersPrices-GEYSER16_7_B1": "Unit16_price",
    "GeysersPrices-GEYSER14_7_N001": "Unit14_price",
    "GeysersPrices-GEYSER11_7_B1": "Unit11_price",
    "GeysersPrices-GEYSER12_7_B1": "Unit12_price",
    "GeysersPrices-GEYSER13_7_N001": "Unit13_price",
    "GeysersPrices-GEOENGY_7_B3": "Aidlin_price"}

#curtailment = pd.ExcelFile("hourly_curtailment.xlsx")
curtailment = pd.ExcelFile("hourly_curtailment_updated.xlsx")
indices_1 = [i for i in range(16)]
indices_2 = [i for i in range(14)]
cur_2013 = pd.read_excel(curtailment, '2013DECs', index_col=0, usecols=indices_1)
cur_2014 = pd.read_excel(curtailment, '2014DECs', index_col=0, usecols=indices_1)
cur_2015 = pd.read_excel(curtailment, '2015DECs', index_col=0, usecols=indices_1)
cur_2016 = pd.read_excel(curtailment, '2016DECs', index_col=0, usecols=indices_2)
cur_2017 = pd.read_excel(curtailment, '2017DECs', index_col=0, usecols=indices_2)
cur_2018 = pd.read_excel(curtailment, '2018DECs', index_col=0, usecols=indices_2)
cur_dfs = [cur_2013, cur_2014, cur_2015, cur_2016, cur_2017, cur_2018]
# smudge = pd.read_csv('GeysersPrices-SMUDGEO1_7_B1.csv', index_col=0, usecols=[6,8])
# smudge.set_index('Local Datetime (Hour Beginning)', inplace=True)

#default join=outer for no information loss
cur = pd.concat(cur_dfs)
#delete NaT index
cur = cur.loc[pd.notnull(cur.index)]
cur = average_duplicates(cur)

for price_node in mapping.keys():
    df = price_dfs[price_node]
    df.rename(columns={price_node: mapping[price_node]}, inplace=True)
    cur = pd.concat([cur,df],axis=1)

cur = pd.concat([cur, sf_avg],axis=1)
cur = pd.concat([cur, u56_avg],axis=1)
cur = pd.concat([cur, u78_avg],axis=1)

#we don't really need rows for which curtailment is null
low = pd.Timestamp('20130101')
high = pd.Timestamp('20190101')
cur = cur[cur.index >= low]
cur = cur[cur.index < high]

cur.to_csv("curtailment_and_prices.csv")

#sample plot
# plot_curtail_and_price("Unit 7&8", "Unit7_8_price", cur)