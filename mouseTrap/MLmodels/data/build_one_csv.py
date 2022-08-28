import os
import pandas as pd
from pandas import DataFrame
from pathlib import Path


def main():
    
    emptyDF=DataFrame()
    time_df=pd.read_csv("one_year_5min_rollups/SACR_SUNN.csv", header=None)
    time_df=time_df.rename(columns={0:"Time",1:"in",2:"out"})
    time_df=time_df.iloc[1:]
    emptyDF['Time']=time_df['Time']


    files=os.listdir('one_year_5min_rollups/')


    for fn in files:
        print(fn)
        fname=Path(fn).name
        fname=fname.replace('.csv','')
        fname_out=fname+"_out"
        fname_in=fname+"_in"
        file_df=pd.read_csv("one_year_5min_rollups/"+fn, header=None)
        file_df=file_df.rename(columns={0:"Time",1:fname_in,2:fname_out})
        file_df=file_df.iloc[1:]
        file_df=file_df.drop(columns=["Time"])
        #print(file_df)
        emptyDF=pd.concat([emptyDF,file_df], axis=1)
        print(emptyDF)

    emptyDF.to_csv("fulldata_5min.csv",sep=',')
        






main()