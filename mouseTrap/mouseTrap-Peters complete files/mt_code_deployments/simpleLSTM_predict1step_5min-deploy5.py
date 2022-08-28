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
from google.cloud import firestore

from google.cloud import storage
from google.cloud import firestore
import cloudstorage as gcs

import json
import glob, os

#import datetime
import calendar

from datetime import datetime
import re
import time
#%matplotlib inline
data_dim=24
batch_size=1
timesteps=8
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
	#print("values")
	#print(Y)
	

	#print("correspondingvalues")
	#data_v = traffic_scaler.inverse_transform(actual_predictions.reshape(-1,1))
	#print(data_v)
	#reshape to [samples, features, timesteps]
	X = X.reshape(X.shape[0], 1, 1)
	Y = Y.reshape(Y.shape[0], 1)
	#Train-test split
	#X_train = X[:105073]
	#Y_train = Y[:105073]
	#X_val = X[105073:105097]
	#Y_val = Y[105073:105097]
	#X_test=X[105097:]
	#Y_test=Y[105097:]
	X_train = X[:100]
	Y_train = Y[:100]
	X_val = X[100:150]
	Y_val = Y[100:150]
	X_test=X[-1:]
	Y_test=Y[-1:]

	#if print_shapes:
	#	print("X_train shape: ", X_train.shape)
	#	print("Y_train shape: ", Y_train.shape)
	#	print("X_val shape: ", X_val.shape)
	#	print("Y_val shape: ", Y_val.shape)
	#	print("X_test shape: ", X_test.shape)
	#	print("Y_test shape: ", Y_test.shape)

	return scaler, traffic_scaler, X_train, Y_train, X_val, Y_val,X_test,Y_test



#################
def train_val_predictions(model, X_train, Y_train, X_val, Y_val,X_test,Y_test,scaler):
	print("2")
	start=time.perf_counter()
	#print("values got")
	#print(X_train.size, Y_train.size, X_val.size, Y_val.size,X_test.size,Y_test.size)
	X_train_pred = model.predict(X_train, batch_size) #X_train_pred_inv = inverse_transform(X_train_pred, scaler)
	X_val_pred = model.predict(X_val, batch_size) #X_val_pred_inv = inverse_transform(X_val_pred, scaler)
	X_test_pred = model.predict(X_test, batch_size)
	####

	#print("VAL")
	#for n in X_val_pred:
		#print(n[0])

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
	for j in range(2): #change from 24 to 2
		newarray=model.predict(newpredictions2, batch_size)
		onevalue=newarray[0]
		newpredictions2[0]=onevalue
		newpredictions.append(onevalue)
		#print("####")
		#print(onevalue)
	print("############")
	#print("new 24 predictions")
	for j in newpredictions:
		print(j[0])
   
	print("----------")
	print("next step ahead")
	#for j in newpredictions:
	print(newpredictions[0][0])
	predicted1value=newpredictions[0][0]
	if predicted1value<0:
		predicted1value=0
	newpredictions = scaler.inverse_transform(newpredictions)
	print(newpredictions)
	end=time.perf_counter()
	print("Time per prediction:")
	diff= end-start
	print(diff) 
	return predicted1value


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

def main():
	training_1yr_dataDF=pd.read_csv('fulldata_5min.csv')
	#df_aofa_lond = pd.read_csv('../trans_atl/lond_newy_out.csv', header=None)
	#df_aofa_lond = df_aofa_lond.rename(columns = {0: "Time", 1:"Out"})
	dates = training_1yr_dataDF['Time'].apply(lambda x: datetime.strptime(x, '%m/%d/%y %H:%M'))


	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1))
	traffic_scaler = MinMaxScaler(feature_range=(0,1))
	#create train, val, test data
	print("4")
	scaler, traffic_scaler, X_train_linkdata, Y_train_linkdata, X_val_linkdata, Y_val_linkdata,X_test_linkdata,Y_test_linkdata = train_test_wfeatures(training_1yr_dataDF[['Time', 'SACR_SUNN_in']], scaler, traffic_scaler)
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
	plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
	EPOCHS=1
	hist=model.fit(X_train_linkdata, Y_train_linkdata, epochs=EPOCHS,
     batch_size=batch_size, shuffle=False, 
     validation_data = (X_val_linkdata, Y_val_linkdata), 
     verbose = 0, callbacks=[time_callback])
	
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
	model.save("model.h5")
	#correcting values
	



def predict_onsaved(srcname, destname, modelname):

	print("loading")
	print(modelname)

	model=load_model(modelname)#'model.h5')#, custom_objects={'lr':learning_rate})

	#model.summary()

	db = firestore.Client()
	users_ref = db.collection(u'5minrollups')
	
	query_ref=users_ref.where(u'src', u'==', srcname).where(u'dest',u'==',destname).stream()
	print("$$$$$$$$$$$$$$")
	#get last timestamp
	last_timestamp=0
	last_traffic=0
	
	training_1yr_dataDF=pd.DataFrame(columns=['timestamp','traffic'])

	for doc in query_ref:
		receivedDict=doc.to_dict()
		#print(u'{} => {}'.format(doc.id,doc.to_dict()))
		training_1yr_dataDF=training_1yr_dataDF.append({'timestamp':receivedDict['timestamp'], 'traffic': receivedDict['traffic']},ignore_index=True)
		if receivedDict['timestamp']>last_timestamp:
			last_timestamp=receivedDict['timestamp']
			last_traffic=receivedDict['traffic']

	print("$$$$$$$$$$$$$$")
	print(last_timestamp)
	print(last_traffic)
	print("$$$$$$$$$$$$$$")
	#print(training_1yr_dataDF)



	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1))
	traffic_scaler = MinMaxScaler(feature_range=(0,1))
	#create train, val, test data
	
	scaler, traffic_scaler, X_train_linkdata, Y_train_linkdata, X_val_linkdata, Y_val_linkdata,X_test_linkdata,Y_test_linkdata = train_test_wfeatures(training_1yr_dataDF, scaler, traffic_scaler)
	print("1")

	# make predictions
	
	x=1
	
	print("updating model.....")
	batch_size=1

	for i in range(1):
		model.fit(X_val_linkdata, Y_val_linkdata, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()

	print("Predicting Again model.....")
	print("SAVING new MODEL")
	formfilename="model-"+ srcname + "--"+ destname+ ".h5"
	justfilename="model-"+ srcname + "--"+ destname+ ".h5"
	print(formfilename)
	model.save(formfilename)

	#upload to bucket
	updatedmodelbucket = 'updated2019models'
	blobnamenewmodel=formfilename
	client = storage.Client()
	cbucket = client.get_bucket(updatedmodelbucket)
	cblob=cbucket.blob(blobnamenewmodel)

	cblob.upload_from_filename(formfilename)
	print("uploaded new model")


	#total_truth_linkdata, total_pred_linkdata, total_pred_test_linkdata = 
	predictedvalueret=train_val_predictions(model, X_train_linkdata, Y_train_linkdata,X_val_linkdata, Y_val_linkdata, X_test_linkdata, Y_test_linkdata,scaler)
	print(type(predictedvalueret))
	predictedvalueret=float(predictedvalueret)
	#if predictedvalueret>=0:
	#	predictedvalueret=predictedvalueret.item()
	#else: 
	#	predictedvalueret=0
	return predictedvalueret


def build_edge_map():
    client = storage.Client()
    globalbucket = client.get_bucket('mousetrap_global_files')
    edgesblob=globalbucket.get_blob('esnet_edges.json')

    edgesblobstr=edgesblob.download_as_string()

    edge_dicts = json.loads(edgesblobstr)
    #print(edge_dicts)
    edges=edge_dicts['data']['mapTopology']['edges']
    
    return edges


def loopingFiveMin():

	start_time=time.time()

	client = storage.Client()
	
	# change name of bucket if ran more than once
	#old models are in bucket original2018models #updated2019models

	globalbucket = client.get_bucket('updated2019models')

	blobs=client.list_blobs('updated2019models')
	
	for blob in blobs:
		print(blob.name)
	#lsvn-sunn
	all_edges=build_edge_map()
	srcname="LSVN"
	destname="SUNN"
	createlocalmodel='localcopy.h5'
	
	#edgepredictiondata=[]

	all5minpredictions=[]

	edgename=srcname+"--"+destname
	createblobname='model-'+edgename+'.h5'

	try:
		modelblob=globalbucket.blob(createblobname)
		print(modelblob)
		modelblob.download_to_filename(createlocalmodel)
		modelname=createlocalmodel
		print("Model found!")
		pv=predict_onsaved(srcname,destname,modelname)
			
	except Exception as e:
		print(e)
		print("Model not found!")
		#print(createblobname)
		pv=0

	#print(e)
	nowtm=datetime.now()
	nowtimestamp=calendar.timegm(nowtm.utctimetuple())
	print(nowtimestamp)
		
	db = firestore.Client()
	tmpstring='lsvn--sunn--'+str(nowtimestamp)
	print("saving to firestore")
	#pv=pv.item()
	print(pv)

	doc_ref = db.collection(u'5minpredictions').document(tmpstring)
	doc_ref.set({
		u'src':u'LSVN',
		u'dest':u'SUNN',
		u'timestamp':nowtimestamp,
		u'predictedtraffic':pv
	})

	#end link 
	#	break
	#nash-wash
	all_edges=build_edge_map()
	srcname="NASH"
	destname="WASH"
	createlocalmodel='localcopy.h5'
	
	#edgepredictiondata=[]

	all5minpredictions=[]

	edgename=srcname+"--"+destname
	createblobname='model-'+edgename+'.h5'

	try:
		modelblob=globalbucket.blob(createblobname)
		print(modelblob)
		modelblob.download_to_filename(createlocalmodel)
		modelname=createlocalmodel
		print("Model found!")
		pv=predict_onsaved(srcname,destname,modelname)
			
	except Exception as e:
		print(e)
		print("Model not found!")
		#print(createblobname)
		pv=0

	#print(e)
	nowtm=datetime.now()
	nowtimestamp=calendar.timegm(nowtm.utctimetuple())
	print(nowtimestamp)
		
	db = firestore.Client()
	tmpstring='nash--wash--'+str(nowtimestamp)
	print("saving to firestore")
	#pv=pv.item()
	print(pv)

	doc_ref = db.collection(u'5minpredictions').document(tmpstring)
	doc_ref.set({
		u'src':u'NASH',
		u'dest':u'WASH',
		u'timestamp':nowtimestamp,
		u'predictedtraffic':pv
	})

	#end link 
	#	break

	#nersc-sunn
	all_edges=build_edge_map()
	srcname="NERSC"
	destname="SUNN"
	createlocalmodel='localcopy.h5'
	
	#edgepredictiondata=[]

	all5minpredictions=[]

	edgename=srcname+"--"+destname
	createblobname='model-'+edgename+'.h5'

	try:
		modelblob=globalbucket.blob(createblobname)
		print(modelblob)
		modelblob.download_to_filename(createlocalmodel)
		modelname=createlocalmodel
		print("Model found!")
		pv=predict_onsaved(srcname,destname,modelname)
			
	except Exception as e:
		print(e)
		print("Model not found!")
		#print(createblobname)
		pv=0

	#print(e)
	nowtm=datetime.now()
	nowtimestamp=calendar.timegm(nowtm.utctimetuple())
	print(nowtimestamp)
		
	db = firestore.Client()
	tmpstring='nersc--sunn--'+str(nowtimestamp)
	print("saving to firestore")
	#pv=pv.item()
	print(pv)

	doc_ref = db.collection(u'5minpredictions').document(tmpstring)
	doc_ref.set({
		u'src':u'NERSC',
		u'dest':u'SUNN',
		u'timestamp':nowtimestamp,
		u'predictedtraffic':pv
	})

	#end link 
	#	break
	#fpppl-wash
	all_edges=build_edge_map()
	srcname="PPPL"
	destname="WASH"
	createlocalmodel='localcopy.h5'
	
	#edgepredictiondata=[]

	all5minpredictions=[]

	edgename=srcname+"--"+destname
	createblobname='model-'+edgename+'.h5'

	try:
		modelblob=globalbucket.blob(createblobname)
		print(modelblob)
		modelblob.download_to_filename(createlocalmodel)
		modelname=createlocalmodel
		print("Model found!")
		pv=predict_onsaved(srcname,destname,modelname)
			
	except Exception as e:
		print(e)
		print("Model not found!")
		#print(createblobname)
		pv=0

	#print(e)
	nowtm=datetime.now()
	nowtimestamp=calendar.timegm(nowtm.utctimetuple())
	print(nowtimestamp)
		
	db = firestore.Client()
	tmpstring='pppl--wash--'+str(nowtimestamp)
	print("saving to firestore")
	#pv=pv.item()
	print(pv)

	doc_ref = db.collection(u'5minpredictions').document(tmpstring)
	doc_ref.set({
		u'src':u'PPPL',
		u'dest':u'WASH',
		u'timestamp':nowtimestamp,
		u'predictedtraffic':pv
	})

	#end link 
	#	break
	#sacr-snll
	all_edges=build_edge_map()
	srcname="SACR"
	destname="SNLL"
	createlocalmodel='localcopy.h5'
	
	#edgepredictiondata=[]

	all5minpredictions=[]

	edgename=srcname+"--"+destname
	createblobname='model-'+edgename+'.h5'

	try:
		modelblob=globalbucket.blob(createblobname)
		print(modelblob)
		modelblob.download_to_filename(createlocalmodel)
		modelname=createlocalmodel
		print("Model found!")
		pv=predict_onsaved(srcname,destname,modelname)
			
	except Exception as e:
		print(e)
		print("Model not found!")
		#print(createblobname)
		pv=0

	#print(e)
	nowtm=datetime.now()
	nowtimestamp=calendar.timegm(nowtm.utctimetuple())
	print(nowtimestamp)
		
	db = firestore.Client()
	tmpstring='sacr--snll--'+str(nowtimestamp)
	print("saving to firestore")
	#pv=pv.item()
	print(pv)

	doc_ref = db.collection(u'5minpredictions').document(tmpstring)
	doc_ref.set({
		u'src':u'SACR',
		u'dest':u'SNLL',
		u'timestamp':nowtimestamp,
		u'predictedtraffic':pv
	})

	#end link 
	#	break

	#slac-sunn
	all_edges=build_edge_map()
	srcname="SLAC"
	destname="SUNN"
	createlocalmodel='localcopy.h5'
	
	#edgepredictiondata=[]

	all5minpredictions=[]

	edgename=srcname+"--"+destname
	createblobname='model-'+edgename+'.h5'

	try:
		modelblob=globalbucket.blob(createblobname)
		print(modelblob)
		modelblob.download_to_filename(createlocalmodel)
		modelname=createlocalmodel
		print("Model found!")
		pv=predict_onsaved(srcname,destname,modelname)
			
	except Exception as e:
		print(e)
		print("Model not found!")
		#print(createblobname)
		pv=0

	#print(e)
	nowtm=datetime.now()
	nowtimestamp=calendar.timegm(nowtm.utctimetuple())
	print(nowtimestamp)
		
	db = firestore.Client()
	tmpstring='SLAC--SUNN--'+str(nowtimestamp)
	print("saving to firestore")
	#pv=pv.item()
	print(pv)

	doc_ref = db.collection(u'5minpredictions').document(tmpstring)
	doc_ref.set({
		u'src':u'SLAC',
		u'dest':u'SUNN',
		u'timestamp':nowtimestamp,
		u'predictedtraffic':pv
	})

	#end link 
	#	break


	end_time=time.time()
	total_time=end_time-start_time
	print("total_time: %s seconds" %total_time)


#main()

#predict_onsaved()
def loopfunc():

	while True:
		loopingFiveMin()
		time.sleep(300)

	
#loopingFiveMin()

loopfunc()