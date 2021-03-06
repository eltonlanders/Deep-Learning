# -*- coding: utf-8 -*-
#pip install MiniSom

"""### Importing the libraries"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""## Importing the dataset"""

dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, -1].values

"""## Feature Scaling"""

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

"""##Training the SOM"""

from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len= 15, sigma= 1.0, learning_rate = 0.5)
#the grid size is 10*10 , input length is the # of features in X, sigma is the 
#neighborhood size   
som.random_weights_init(X)   #initializing the weights
som.train_random(data = X, num_iteration = 100)

"""##Visualizing the results"""

from pylab import bone, pcolor, colorbar, plot, show
bone() #initializing the window
pcolor(som.distance_map().T) #returns transpose of the matrix of the distance
#of all the winning nodes
colorbar() #adding the legend
markers = ['o', 's'] #circles and squares
colors = ['r', 'g'] #red if customer did't get approval and vice versa
for i, x in enumerate(X): # i is the # of rows, x is the vector of the rows
    w = som.winner(x) #getting the winning nodes
    plot(w[0] + 0.5, #to put the marker in the center of the square of the SOM
         w[1] + 0.5,
         markers[y[i]], #y contains the labels if customers got approval or no
         markeredgecolor = colors[y[i]], #for each customer, adding the color
         #only coloring the edge of the marker
         markerfacecolor = 'None', #inside color of the marker
         markersize = 10, #size of the marker
         markeredgewidth = 2)
show()

"""## Finding the frauds"""

mappings = som.win_map(X) #mapping for all the different nodes in the SOM
frauds = np.concatenate((mappings[(2,1)], mappings[(5,2)]), axis = 0)
#change the mappings coordinates accordingly to the white squares
#first is x and second is y. The first box is where it starts
#check if the values given to the mappings are present in it or not
frauds = sc.inverse_transform(frauds)

"""##Printing the Fraunch Clients"""

print('Fraud Customer IDs')
for i in frauds[:, 0]:
  print(int(i))
  
"""
The MID(Mean inter neuron distance) is the mean of the distances between 
neurons in a given neighborhood defined by sigma.
The greater the MID is the more is the chance it is an outlier and a fraud
The larger the MID the closer to white colour
Mappings is a dictionary and the keys contain the coordinates of the winning
nodes, size is the number of customers associated with that winning node
"""
