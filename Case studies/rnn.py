# Recurrent Neural Network


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values 

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler   #normalisation instead of standardisation for an RNN when it has sigmoid function in the output layer
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output   #timesteps are the previous financial days data the model takes into account for the next prediction
X_train = []   #for every financial day, this will contain the 60 previous financial days data
y_train = []   #this will contain the next financial day data
for i in range(60, 1258):   # i-60 is the first i value 
    X_train.append(training_set_scaled[i-60:i, 0])   #first i will be 60 and it will append all the stock prices till 60 i.e. 59 as upper bound is excluded
    y_train.append(training_set_scaled[i, 0]) 
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))   #shape[0] and [1] gives the # of observations and the columns respectively. This is a 3D tensor which has first dimension as the # of stock prices and then the # of timestamps and finaaly a new diension
#which is called the indicators which gives the additional # of indicators of the stock price  


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential() 

# Adding the first LSTM layer and some Dropout regularisation   #dropout layer is added to avoid overfitting
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))   #units are LSTM units or neurons, here 50 units in the 1st LSTM layers and so on. Return_seq is True coz we are adding more LSTM layers(on the last one we will set it to False)
#for the shape we only take the second and the last diemnsions into account coz the first one i.e. of the observations is automatically taken into account 
regressor.add(Dropout(0.2))   #the rate of neuron that has to be ignored or drop in the layers, here dropping 20% of neurons in the layer(10 out of 50 will be ignored)

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))   #input shape to be specified only for the first layer
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))   #output has only one dimension as it is a real value hence 1 neuron . units corresponds to the # of neurons

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')   #can use RMSprop and MSE for regression

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)   #instead of updating the weights for every observation here it will be done on the batch size of 32


# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)   #the original datasets are concatenated. 0 for vertical
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values   #these are the inputs we need to predict the stock price of jan 2017.
#Here len(dataset_total) - len(dataset_test) - 60 gives the first financial day of jan i.e. 3rd and a colon is added to get the second last day i.e. is the upper bound coz we need the previous 60 days till here to predict the last day
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs) 
X_test = []
for i in range(60, 80):   #20 test data and 60 previous days data
    X_test.append(inputs[i-60:i, 0])   #inputs scaled
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))   #3D structure
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)   #unscale the data

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')   #20 days data 
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

"""
There are 20 days in a financial month(sat-sun not included)

Here a stacked LSTM model is built which is robust and has a lot of layers


"""
