# stock-price-prediction-lstm
GOOG Dataset description:The data used in this project is of the Alphabet Inc3 from January 1, 2005 to June 20,
2017 , this is a series of data points indexed in time order or a time series. My goal was to
predict the closing price for any given date after training. For ease of reproducibility and
reusability, all data was pulled from the Google Finance Python API


MSFT dataset description:
Dataset is downloaded from https://www.alphavantage.co/documentation/ site for microsoft stock prices. The dataset contains daily opening,
closing, high and low stock prices. The dataset contains entries from 2000-01-03 to 2018-06-20.

Training model on google dataset:
1. comment line no. 10 of utils.py file and make sure line no. 9 is uncommnet.
2. uncomment line no. 13, 23 and 24 of model.py file 
3. run the model.py file using below command
python model.py

Training model on msft dataset:
1. comment line no. 9 of utils.py file and make sure line no. 10 is uncommnet.
2. change the text written on line no. 13 of model.py file from 'scale_google.pkl' to 'scale_msft.pkl', similarlry on line no. 23 from "google.hdf5" to "msft.hdf5" and same on line no. 25
3. run the model.py file using below command
python model.py

Q. How training data is created?
Ans: X_train = [[0.01964948]
 [0.01383108]
 [0.01312325]
 [0.00961239]
 [0.01336391]]

Y_train = [[0.01383108]
 [0.01312325]
 [0.00961239]
 [0.01336391]
 [0.01422747]]

The feature set(X_train) is lagged version of target set(Y_train). The above values are scaled version of original stock values. This is done because while predicting the closing price of next day we must know the prior information about the closing price i.e closing price of current day.


Q. how train test split is done?
Normally we shuffle the data to split into train and test but this is a time series data where current value depend upon the previous value so instead of shuffle, i have used first 90% of the data as training data and remaining 10% data as testing data.
Also while training time series data shuffle argument in models fit method was kept to False because sequence of stock prices matters.


Evluation metric:
The predictions are evaluated using rmse(root mean square error.) The ideal value is zero for rmse.


Notebook file:
1. google_model.ipynb
2. msft_model.ipynb
3. forecast.ipynb - predicts the closing price for any given date after training

These are self explanatory files.

keras, sklearn & tensorflow versions:
Keras==2.2.0
scikit-learn==0.19.1
tensorflow==1.8.0


Note:
The one improvement could be implementation of online learning(continously updating the dataset as well as updating the model for new entries) for better future closing price predictions
