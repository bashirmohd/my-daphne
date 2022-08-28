from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM, Activation, Dropout, Bidirectional, TimeDistributed, RepeatVector
#from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD
from random import uniform
import json
# Fix AttributeError: 'module' object has no attribute 'control_flow_ops'
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
tensorflow.control_flow_ops = control_flow_ops

tf.Session(config=tf.ConfigProto(log_device_placement=True))


def inverse_transform(pred, traffic_scaler):
	return traffic_scaler.inverse_transform(pred)

def graph_results(model, X_test, Y_test, traffic_scaler, batch_size = 1):
    # walk-forward validation on the test data
    pred_x_test = model.predict(X_test, batch_size)
    pred_test = inverse_transform(pred_x_test, traffic_scaler)
    y_test = np.float_(Y_test)
    y_test_inv = inverse_transform(y_test, traffic_scaler)

    line_test_pred = np.reshape(pred_test, pred_test.shape[0])
    line_test_real = np.reshape(y_test_inv, y_test_inv.shape[0])


def train_test_wfeatures(df, pathway, split_proportion, scaler, traffic_scaler, print_shapes = True):
	X = df.drop('Time', axis = 1).as_matrix() #drop time to get all features
	Y = df[[pathway]].shift(1).fillna(0).as_matrix() #shift traffic values down 1 to create response variable

    #Normalize
	X = scaler.fit_transform(X)
	Y = traffic_scaler.fit_transform(Y)

    #reshape to [samples, features, timesteps]
	X = X.reshape(X.shape[0], 1, X.shape[1])

    #Train-test split
	row = int(round(split_proportion * df.shape[0]))
	X_train = X[:row]
	Y_train = Y[:row]
	X_test = X[row:]
	Y_test = Y[row:]
    
	if print_shapes:
		print("X_train shape: ", X_train.shape)
		print("Y_train shape: ", Y_train.shape)
		print("X_test shape: ", X_test.shape)
		print("Y_test shape: ", Y_test.shape)

	return X_train, Y_train, X_test, Y_test

scaler = MinMaxScaler(feature_range=(0,1))
traffic_scaler = MinMaxScaler(feature_range=(0,1))
X_train, Y_train, X_test, Y_test = train_test_wfeatures(points_1767_chicstar, 
                                                        "CHIC--STAR", 0.9, scaler, traffic_scaler)

nb_epoch = 20

print("-- Building --")
model.compile(loss="mse",
                  optimizer = 'adam')

print('-- Training --')
for i in range(nb_epoch):
    model.fit(X_train, Y_train, nb_epoch=1, batch_size=batch_size, shuffle=False)
    model.reset_states()


score = model.evaluate(X_test, Y_test, verbose=0, batch_size = 1)
print('Test loss:', score)

graph_results(model, X_test, Y_test, traffic_scaler)