# gait study Neural Network main entry - Derek Liu 2017/12/08

from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from keras import optimizers
import numpy
import pandas
numpy.random.seed(7)

#load data

data = pandas.read_csv("D:\\Derek\\Matlab\\gait_study\\algorithm\\train\\data\\SubFeaturesMedian_stage2.csv",delimiter=',',header=0)
data_set = data.values
X = data_set[:,[1,2,3,5]]
Y = data_set[:,6]
print("data loaded, length = %d, proceed to define model" %(len(data_set[:,0])))

#define model
model = Sequential()
num_neurons_1st = 10
num_neurons_2nd = 10
num_input = len(X[0,:])
model.add(Dense(num_neurons_1st, input_dim= num_input, activation='sigmoid')) #try sigmoid later
model.add(Dense(num_neurons_2nd, input_dim= num_neurons_1st, activation='sigmoid'))
model.add(Dense(1,input_dim=num_neurons_2nd,activation='sigmoid'))
#compile model with adam (gradient descent)
model.compile(loss= 'mape', optimizer = 'sgd', metrics=['mape'])

#fit model with input and output
history = model.fit(X, Y, epochs=1000, batch_size=100)
min_error = min(history.history['mean_absolute_percentage_error'])
print("train complete, minimal mape: %.2f%%" %(min_error))
pyplot.plot(history.history['mean_absolute_percentage_error'])
pyplot.show()