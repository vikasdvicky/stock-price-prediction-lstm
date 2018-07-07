from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import dateparser
import numpy as np

np.random.seed(100) #for reproducibility

def read_data():
    data = pd.read_csv('google_dataset.csv')
    # data = pd.read_csv('daily_MSFT.csv')
    return data['Close'][::-1] #reversing the stock prices from previous date to till date

def create_XY(data): #converting the univariate time series into multivariate time series(making it supervised learning)
    X = data
    Y = X[1:]
    X = X[0:-1]
    return X,Y

def scale_data(data): #scaling data between 0 and 1 because LSTMs are sensitive to the scale of the input data.
    scale = MinMaxScaler(feature_range=(0, 1))
    scale = scale.fit(np.array(data).reshape(-1,1))
    return scale

def get_train_test(train_size=0.9): #dividing the dataset into train and test set. Default train size is 90%
    target = read_data()
    train, test = target[0:int(train_size*len(target))], target[int(train_size*len(target)):]
    scale = scale_data(train) #fir the scaler model on train data
    train = scale.transform(train.reshape(-1,1)) #scaling train set
    test = scale.transform(test.reshape(-1,1)) #scaling test set
    x_train, y_train = create_XY(train) # creating supervised train data
    x_test, y_test = create_XY(test) # creating supervised test data
    return x_train, y_train, x_test, y_test, scale