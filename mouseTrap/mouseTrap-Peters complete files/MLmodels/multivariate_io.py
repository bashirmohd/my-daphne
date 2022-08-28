#using the stacked lstm model

#stacked 2 LSTMs
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


from datetime import datetime
import re
import time
import os
#%matplotlib inline
data_dim=24
batch_size=1
timesteps=8
#univariate feature
n_features=1



class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


def train_test_wfeatures(df, scaler, traffic_scaler, print_shapes = True):
	"""Returns training and test data from Pandas DataFrame of data"""
	print("3")
	#Split features from response variable
	#X = df.iloc[:,1].values #dropping time
	#Y = df.iloc[:,1].shift(1).fillna(0).values #shift traffic values  1 to create response variable
	X=df
	Y=X.shift(periods=1,fill_value=0)
	print(X.shape)
	print(Y.shape)
	#Normalize
	X = scaler.fit_transform(X) #.reshape(-1,1))
	Y = traffic_scaler.fit_transform(Y) #.reshape(-1,1))
	print(X)
	print("values")
	print(Y)
	#print(Y.shape)
	#c=np.array(Y)
	#np.savetxt("real-Y.csv", c, delimiter=',')




    #reshape to [samples, features, timesteps]


	#X = X.reshape(X, 34, 1)
	#Y = Y.reshape(Y, 34)
	#Train-test split
	X_train = X[:8688]
	Y_train = Y[:8688]
	X_val = X[8688:8712]
	Y_val = Y[8688:8712]
	X_test=X[8712:]
	Y_test=Y[8712:]

	if print_shapes:
		print("X_train shape: ", X_train.shape)
		print("Y_train shape: ", Y_train.shape)
		print("X_val shape: ", X_val.shape)
		print("Y_val shape: ", Y_val.shape)
		print("X_test shape: ", X_test.shape)
		print("Y_test shape: ", Y_test.shape)

	return scaler, traffic_scaler, X_train, Y_train, X_val, Y_val,X_test,Y_test



def main():
	#prepare the dataset
	training_1yr_data1 = os.path.join('data/trans_1Y_part1.txt')
	training_1yr_data1DF=pd.read_csv(training_1yr_data1)
    #drop date column
	training_1yr_data1DF=training_1yr_data1DF.drop(columns='Days')


	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1))
	traffic_scaler = MinMaxScaler(feature_range=(0,1))
	#create train, val, test data
	print("4")
	#scaler, traffic_scaler, X_train_linkdata, Y_train_linkdata, X_val_linkdata, Y_val_linkdata,X_test_linkdata,Y_test_linkdata = train_test_wfeatures(training_1yr_data1DF, scaler, traffic_scaler)
	print("1")

	time_callback=TimeHistory()

	X = scaler.fit_transform(training_1yr_data1DF) #.reshape(-1,1))

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

	train=values[:no_training_hours,:]
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


	# define create and fit the LSTM network - 2 layer
    #setting bacth size to 1 online training, improve accuracy
	n_batch=1 
	n_epoch=5
	n_neurons=68

	#build mode;
	model = Sequential()
	#return 24 vector points
	print("3") 
	model.add(LSTM(n_neurons, activation='tanh',batch_input_shape=(n_batch, train_X.shape[1], train_X.shape[2]),stateful=True))
	print("2")
	#X_train_linkdata.shape[1],X_train_linkdata.shape[2]), stateful=True))
	#model.add(Dropout(0.5))

	#model.add(LSTM(34, activation='tanh',return_sequences=True))
	#model.add(Dropout(0.5))
	print("5")

	#model.add(Flatten())

	model.add(Dense(34)) #fitting to one class output predicting t+1
	model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
	
	print(model.summary())
	plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

	for i in range(n_epoch):
		hist=model.fit(train_X, train_y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
		hist=model.reset_states()	

	#model.reset_states()
	#forecasting on validation data

	for i in range(len(test_X)):
		#test_X=test_X.reshape((test_X.shape[0],1, test_X.shape[1]))
		print(test_X)
		print(test_X.shape)
		yhat=model.predict(test_X,batch_size=1)
		print(yhat)


	#print(hist.history)#outputs val_loss, val_acc, acciuracy
	#print(hist.epoch)#number of epochs 10
	print("epoch times:")
	print(time_callback.times) #time per epoch

	sumt=0
	for t in time_callback.times:
		sumt=sumt+t
	sumt=sumt/10
	print("Avergae time per epoch:")
	print(sumt)

	# make predictions

	graph_results(model,X_val_linkdata, Y_val_linkdata, X_test_linkdata, Y_test_linkdata, traffic_scaler)

	total_truth_linkdata, total_pred_linkdata, total_pred_test_linkdata = train_val_predictions(model, X_train_linkdata, Y_train_linkdata,X_val_linkdata, Y_val_linkdata, X_test_linkdata, Y_test_linkdata)
	x=1
	print("6")

	# walk-forward validation on the test data
	line_test_pred = np.reshape(total_pred_linkdata[x:], total_pred_linkdata[x:].shape[0])
	line_test_real = np.reshape(total_truth_linkdata[x:], total_truth_linkdata[x:].shape[0])
	line_test_real_withtest = np.reshape(total_pred_test_linkdata[x:], total_pred_test_linkdata[x:].shape[0])

	print(line_test_pred.size,line_test_real.size,line_test_real_withtest.size)
	print("7")
	a=np.array([line_test_pred,line_test_real,line_test_real_withtest]).T
	np.savetxt("foo.csv", a, delimiter=',')


	actual_predictions=line_test_real_withtest[0:]
	print(actual_predictions)
	#print("correspondingvalues")
	#data_v = traffic_scaler.inverse_transform(actual_predictions.reshape(-1,1))
	#print(data_v)
	b=np.array(actual_predictions)
	np.savetxt("real2.csv", b, delimiter=',')
	#dates for plot
	#dates = df_sacr_denv['Time'].apply(lambda x: datetime.fromtimestamp(x/1000.))
	dates = df_aofa_lond['Time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

	moreX = df_aofa_lond[['Time', 'Out']].iloc[:,1].values
	moreX_add = moreX[2136:]
	line_test_real=np.append(line_test_real,moreX_add)

	fig1 = plt.figure(figsize=(20,10))
	plt.plot(dates[x:], line_test_real, color='blue',label='Original', linewidth=1)
	print(line_test_real.size)
	print("5")

	line_test_real_withtest=np.append(line_test_real_withtest,moreX_add)
	plt.plot(dates[x:], line_test_real_withtest, color='red',label='Prediction', linewidth=1)
	plt.axvline(x=dates[2136], color = 'green')
	plt.legend(loc='best')
	plt.title('Predictions')


main()