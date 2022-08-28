#simple 1 lstm model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.utils.vis_utils import plot_model
from keras.layers import Activation, Dropout
from keras.models import load_model


from datetime import datetime
import re
import time
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


def train_test_wfeatures(df, scaler, traffic_scaler, print_shapes = True):
	"""Returns training and test data from Pandas DataFrame of data"""
	print("3")
	#Split features from response variable
	X = df.iloc[:,1].values #dropping time
	Y = df.iloc[:,1].shift(1).fillna(0).values #shift traffic values  1 to create response variable

	#Normalize
	X = scaler.fit_transform(X.reshape(-1,1))
	Y = traffic_scaler.fit_transform(Y.reshape(-1,1))
	print("values")
	print(Y)
	c=np.array(Y)
	#np.savetxt("real-Y1.csv", c, delimiter=',')
	#dates for plot

	#print("correspondingvalues")
	#data_v = traffic_scaler.inverse_transform(actual_predictions.reshape(-1,1))
	#print(data_v)
	#reshape to [samples, features, timesteps]
	X = X.reshape(X.shape[0], 1, 1)
	Y = Y.reshape(Y.shape[0], 1)
	#Train-test split
	X_train = X[:105073]
	Y_train = Y[:105073]
	X_val = X[105073:105097]
	Y_val = Y[105073:105097]
	X_test=X[105097:]
	Y_test=Y[105097:]

	if print_shapes:
		print("X_train shape: ", X_train.shape)
		print("Y_train shape: ", Y_train.shape)
		print("X_val shape: ", X_val.shape)
		print("Y_val shape: ", Y_val.shape)
		print("X_test shape: ", X_test.shape)
		print("Y_test shape: ", Y_test.shape)

	return scaler, traffic_scaler, X_train, Y_train, X_val, Y_val,X_test,Y_test



#################
def train_val_predictions(model, X_train, Y_train, X_val, Y_val,X_test,Y_test,scaler):
	print("2")
	start=time.perf_counter()
	print("values got")
	print(X_train.size, Y_train.size, X_val.size, Y_val.size,X_test.size,Y_test.size)
	X_train_pred = model.predict(X_train, batch_size) #X_train_pred_inv = inverse_transform(X_train_pred, scaler)
	X_val_pred = model.predict(X_val, batch_size) #X_val_pred_inv = inverse_transform(X_val_pred, scaler)
	X_test_pred = model.predict(X_test, batch_size)
	####

	print("VAL")
	for n in X_val_pred:
		print(n[0])

	print("XTEST")
	print(X_test_pred)
	onevalue=X_test_pred[0]
	newpredictions=[]
	newpredictions.append(onevalue)
	newpredictions2=X_test
	newpredictions2[0]=onevalue
	#newpredictions2=onevalue
	#newpredictions2[1]=onevalue

	#j=0
	for j in range(24):
		newarray=model.predict(newpredictions2, batch_size)
		onevalue=newarray[0]
		newpredictions2[0]=onevalue
		newpredictions.append(onevalue)
		#print("####")
		#print(onevalue)
	print("############")
	print("new 24 predictions")
	for j in newpredictions:
		print(j[0])

	print("----------")

	newpredictions = scaler.inverse_transform(newpredictions)
	print(newpredictions)


	end=time.perf_counter()


	y_train = np.float_(Y_train)#y_train_inv = inverse_transform(y_train, scaler)
	y_val = np.float_(Y_val)#y_val_inv = inverse_transform(y_val, traffic_scaler)
	y_test = np.float_(Y_test)


	total_truth = np.vstack((y_train, y_val))
	total_pred = np.vstack((X_train_pred, X_val_pred))
	total_pred_test = np.vstack((X_train_pred, X_test_pred))
	print("Time per prediction:")

	diff= end-start
	print(diff)


	print(total_truth.size,total_pred.size, total_pred_test.size)

	return total_truth, total_pred, total_pred_test

# load the dataset
#dataframe = pandas.read_csv('airline-passengers.csv', usecols=[1], engine='python')
#dataset = dataframe.values
#dataset = dataset.astype('float32')

####

def inverse_transform(pred, traffic_scaler):
    return traffic_scaler.inverse_transform(pred)

def graph_results(model,X_val, Y_val, X_test, Y_test, traffic_scaler, batch_size = 1):
    # walk-forward validation on the test data
    pred_x_test = model.predict(X_test, batch_size)
    pred_test = inverse_transform(pred_x_test, traffic_scaler)

    y_test = np.float_(Y_test)
    y_test_inv = inverse_transform(y_test, traffic_scaler)

    line_test_pred = np.reshape(pred_test, pred_test.shape[0])
    line_test_real = np.reshape(y_test_inv, y_test_inv.shape[0])
    plt.figure(figsize=(20,10))
    plt.plot(line_test_real, color='blue',label='Original', linewidth=1)
    plt.plot(line_test_pred, color='red',label='Prediction', linewidth=1)
    plt.legend(loc='best')
    plt.title('Test - Comparison')
    plt.show()

#####

def main(response, target, training_1yr_dataDF):
	#training_1yr_dataDF=pd.read_csv('fulldata_5min.csv')
	#df_aofa_lond = pd.read_csv('../trans_atl/lond_newy_out.csv', header=None)
	#df_aofa_lond = df_aofa_lond.rename(columns = {0: "Time", 1:"Out"})
	dates = training_1yr_dataDF[response].apply(lambda x: datetime.strptime(x, '%m/%d/%y %H:%M'))


	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1))
	traffic_scaler = MinMaxScaler(feature_range=(0,1))
	#create train, val, test data
	print("4")
	scaler, traffic_scaler, X_train_linkdata, Y_train_linkdata, X_val_linkdata, Y_val_linkdata,X_test_linkdata,Y_test_linkdata = train_test_wfeatures(training_1yr_dataDF[[response, target]], scaler, traffic_scaler)
	print("1")

	# create and fit the LSTM network - 1 layer
	#build mode;
	trainstart=time.perf_counter()

	time_callback=TimeHistory()

	model = Sequential()
	#return 24 vector points
	model.add(LSTM(24, activation='tanh', return_sequences=True, batch_input_shape=(batch_size, X_train_linkdata.shape[1],X_train_linkdata.shape[2]), stateful=True))
	#model.add(Dropout(0.5))
	model.add(Flatten())

	model.add(Dense(1)) #fitting to one class output predicting t+1
	model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])

	print(model.summary())
	plot_model(model, to_file='model_plot_{}.png'.format(target), show_shapes=True, show_layer_names=True)
	EPOCHS=1
	hist=model.fit(X_train_linkdata, Y_train_linkdata, epochs=EPOCHS,
     batch_size=batch_size, shuffle=False,
     validation_data = (X_val_linkdata, Y_val_linkdata),
     verbose = 0, callbacks=[time_callback])

	#model.reset_states()


	#print(hist.history)#outputs val_loss, val_acc, acciuracy
	#print(hist.epoch)#number of epochs 10
	print("epoch times:")
	print(time_callback.times) #time per epoch

	sumt=0
	for t in time_callback.times:
		sumt=sumt+t
	sumt=sumt/EPOCHS
	print("Avergae time per epoch:")
	print(sumt)
	trainend= time.perf_counter()

	trainend=trainend-trainstart
	print("training time")
	print(trainend)

	print("SAVIING MODEL")
	print(model.save("model-{}.h5".format(target)))
	#correcting values
	#graph_results(model,X_val_linkdata, Y_val_linkdata, X_test_linkdata, Y_test_linkdata, traffic_scaler)

######################THIS IS THE LOOP I ADDED#############
data = pd.read_csv('main_fulldata_5min_original.csv')
data.fillna(data.mean(), inplace=True)
features = [i for i in data.columns if i != 'Time']
for feature in features:
    main(response = 'Time', target = feature, training_1yr_dataDF = data)

#predict_onsaved()
