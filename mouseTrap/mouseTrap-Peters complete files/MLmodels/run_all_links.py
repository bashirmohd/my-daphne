#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from pandas import concat
import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.utils.vis_utils import plot_model
from keras.layers import Activation, Dropout
from matplotlib import pyplot
from numpy import concatenate
from math import sqrt
import requests


from datetime import datetime
import re
import time
import os



class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


#create try catch for all links
def main():
    #prepare the dataset
	training_1yr_data = os.path.join('data/one_year_data_34links.txt')
	training_1yr_dataDF=pd.read_csv(training_1yr_data)
	training_1yr_dataDF=training_1yr_dataDF.drop(columns='Days')

    #call each ML model with date and link details
	
	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1))
	traffic_scaler = MinMaxScaler(feature_range=(0,1))
	#create train, val, test data
	print("4")
	#scaler, traffic_scaler, X_train_linkdata, Y_train_linkdata, X_val_linkdata, Y_val_linkdata,X_test_linkdata,Y_test_linkdata = train_test_wfeatures(training_1yr_data1DF, scaler, traffic_scaler)
	print("1")

	time_callback=TimeHistory()

	X = scaler.fit_transform(training_1yr_dataDF) #.reshape(-1,1))

	#reframing problem # Might not need this
	reframed=series_to_supervised(X,1,1)
	print(reframed.head())
	#print(reframed['var1(t-1)'])
	#print(reframed['var1(t)'])

    #split the data set now
    #Train-test split
	values= reframed.values
	no_training_hours=8688
	no_validation_hours=8712

	train=values[:no_training_hours,1]
	print(train)
	val=values[no_training_hours:no_validation_hours,:]
	test=values[no_training_hours:,:]
	print(test)

	train_X, train_y=train[:,:-34],train[:,34:]
	print("train_X")
	print(train_X)
	print("train y")
	print(train_y)

	print("train_X shape: ", train_X.shape)
	print("train_y shape: ", train_y.shape)
    #val and test data
	val_X, val_y=val[:,:-34],val[:,34:]
	test_X,test_y=test[:,:-34],val[:,34:]

	#reshpae input to be 3D [samples, timesteps, features]
	train_X=train_X.reshape((train_X.shape[0],1, train_X.shape[1]))
	print("train_X shape: ", train_X.shape)




main()