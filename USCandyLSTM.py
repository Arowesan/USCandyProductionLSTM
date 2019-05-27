"""
@file: USCandyLSTM.py
@language: Python 3.6.6
@author: Anthony Leung (leungant@yorku.ca)
"""

from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.regularizers import l2

import numpy as np
import scipy.stats as st
import pandas as pd


##-----------------------------------------------
##--------- AUXILLARY FUNCTIONS -----------------
##-----------------------------------------------
def to_stationary_all(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.DataFrame(diff)


def inverse_stationary_all(history, yhat, interval=1):
    inverted = list()
    for i in range(0,len(yhat)):
        value = yhat[i] + history[-(len(history)-i)]
        inverted.append(value)
    return pd.DataFrame(inverted)


def inverse_stationary(history, yhat, interval=1):
    return yhat + history[-interval]


def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(train)
    # Transform train set
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # Transform testset
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


def inverse_scale(scaler, X, y):
    new_row = [x for x in X] + [y]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0,-1]


def lstm_forecast(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]


#-- Parse Data
df = pd.read_csv('candy_production.csv', header=0, index_col=0, squeeze=True)
raw_values = df.values
raw_keys = df.keys()

#-- Transform time series to stationary
raw_values_stationary = to_stationary_all(raw_values, 1)

#-- Create supervised learning time series
supervised_set = timeseries_to_supervised(raw_values_stationary, lag=1)
supervised_val = supervised_set.values

#-- Split into train and test sets (ratio of 7:3)
splitIdx = int(len(supervised_val) * 0.7)

train, test = supervised_val[0:splitIdx], supervised_val[splitIdx:]

#-- Scale the data
scaler, train_scaled, test_scaled = scale(train, test)

#-- Initialize parameters for LSTM model
look_back       = 1
batch_size      = 1
lbd             = 1e-4    # For l2 regularization
p_drop          = 0.5     # dropout rate for LSTM layer, dropout
p_dropRecurrent = 0.25    # dropout rate for LSTM layer, recurrent dropout
p_dropDense     = 0.1     # dropout rate for Dense layer

#-- Separate sets into X and y components
Xtrain, ytrain = train_scaled[:,0:-1], train_scaled[:,-1]
Xtest, ytest = test_scaled[:,0:-1], test_scaled[:,-1]

#-- Reshape X sets for LSTM model
Xtrain = Xtrain.reshape(Xtrain.shape[0], 1, Xtrain.shape[1])
Xtest = Xtest.reshape(Xtest.shape[0], 1, Xtest.shape[1])

#-- Construct LSTM model
model = Sequential()
model.add(LSTM(4, batch_input_shape=(batch_size, look_back,1), 
               dropout=p_drop, recurrent_dropout=p_dropRecurrent,
               stateful=True, kernel_regularizer=l2(lbd)))
model.add(Dropout(p_dropDense))
model.add(Dense(1, kernel_regularizer=l2(lbd)))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

#-- Train the model
model.fit(Xtrain,ytrain, epochs=25, batch_size=batch_size, 
          verbose=2, shuffle=False)

#-- Define backend function for Monte Carlo (MC) Dropout
predict_stochastic = K.function([model.layers[0].input, K.learning_phase()], 
                                 [model.layers[-1].output])

#-- Use test set to predict with MC dropout
T       = 1000         # Number of iterations for MC Dropout
lphase  = 1            # Learning phase (0 = testing, 1 = training)
yhat_mc = np.array([predict_stochastic([Xtest, lphase]) for _ in range(T)])
yhat_mc = yhat_mc.reshape(-1, ytest.shape[0]).T

#-- Look at one of the predictions
monthIdx   = 0            # Must be less than len(test) = 165
monthStr   = raw_keys[len(train) + monthIdx]
y_expected = raw_values[len(train) + monthIdx]
yhat_test  = yhat_mc[monthIdx,:]
yhat       = []
#-- Reverts predicted value back to the trend from original data
for i in range(0, len(yhat_test)):
    X = test_scaled[monthIdx, 0:-1]
    y = yhat_test[i]
    y = inverse_scale(scaler, X, y)
    y = inverse_stationary(raw_values, y, len(test)+1-monthIdx)
    yhat.append(y)

#-- Compute statistics
yhat_mc_mean       = np.mean(yhat)
yhat_mse           = np.mean((yhat-yhat_mc_mean)**2.0)
yhat_rmse          = (yhat_mse) ** 0.5
confidence         = 0.95
CI_lower, CI_upper = st.norm.interval(confidence, loc = yhat_mc_mean, 
                                      scale = yhat_rmse)
print('\nMC Dropout prediction at %s' % (monthStr))
print('y_expected: %f' % (y_expected))
print('y_predicted: %f' % (yhat_mc_mean))
print('MSE: %f' % (yhat_mse))
print('RMSE: %f' % (yhat_rmse))
print('95%% CI: (%f, %f)\n' % (CI_lower, CI_upper))

input("Press Enter to exit")
