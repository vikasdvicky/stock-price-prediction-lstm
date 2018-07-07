from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from utils import get_train_test
from sklearn.externals import joblib
import numpy as np
import pandas as pd

x_train, y_train, x_test, y_test, scale = get_train_test() # train test split of dataset
# joblib.dump(scale, 'scale_models/scale_google.pkl') 

x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1])) # reshaping data for LSTM
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1])) # reshaping data for LSTM


model = Sequential()
model.add(LSTM(4, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# callback = ModelCheckpoint("weights/google.hdf5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)
# model.fit(x_train, y_train, shuffle=False,epochs=100, batch_size=1, verbose=2, callbacks=[callback], validation_data=(x_test,y_test)) #shuffle should be set to False in time series prediction problem because sequence of stock prices matters .
model.load_weights('weights/google.hdf5')

test = model.predict(x_test)
y_true = scale.inverse_transform(y_test).ravel() # inverse scaling of the original test data. inverse scaling is done to get the original stock value which was converted into 0 and 1 range
y_pred = scale.inverse_transform(test).ravel() # inverse scaling of the predicted data
plot = pd.DataFrame(np.stack([y_true,y_pred],axis=1), columns=["True","Predicted"]) #creating dataframe of true and predicted values for ploting
plot.plot()
pyplot.show()
err = np.sqrt(mean_squared_error(y_true, y_pred)) #calculation of root mean square error
print "Root Mean Squared Error:",err
from keras.utils import plot_model
plot_model(model, to_file='model.png')