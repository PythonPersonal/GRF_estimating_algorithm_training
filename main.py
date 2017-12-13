# gait study Neural Network main entry - Derek Liu 2017/12/08

from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from datetime import datetime
from findAllIndicesInList import all_indices
import numpy
import pandas
numpy.random.seed(27)

# load data

data = pandas.read_csv("D:\\Derek\\Matlab\\gait_study\\algorithm\\train\\data\\SubFeatures_stage2.csv",
                       delimiter=',', header=0)
data_set = data.values
SubIdxForValidate = [4, 22, 6, 11, 21]
allIndicesToValidate = []
for idcs in SubIdxForValidate:
    temp_idcs = all_indices(idcs, list(data_set[:, 0]))
    allIndicesToValidate = allIndicesToValidate + temp_idcs
allIndicesToValidate.sort()
allIndices = numpy.arange(0, len(data_set[:, 0])-1)
allIndicesToTrain = list(set(allIndices) - set(allIndicesToValidate))

X = data_set[:, [1, 2, 3, 5]]
Y = data_set[:, 6]

X_Tr = (data_set[:, [1, 2, 3, 5]])[allIndicesToTrain, :]
Y_Tr = (data_set[:, 6])[allIndicesToTrain]

X_val = (data_set[:, [1, 2, 3, 5]])[allIndicesToValidate, :]
Y_val = (data_set[:, 6])[allIndicesToValidate]
print("data loaded, length = %d, proceed to define model" % (len(data_set[:, 0])))

# define model
model = Sequential()
num_neurons_1st = 10
num_neurons_2nd = 10
optim = 'adam'
b_size = 32
num_input = len(X[0, :])
model.add(Dense(num_neurons_1st, input_dim=num_input, activation='sigmoid'))  # try sigmoid later
model.add(Dense(num_neurons_2nd, input_dim=num_neurons_1st, activation='sigmoid'))
model.add(Dense(1, input_dim=num_neurons_2nd, activation='sigmoid'))
# compile model with adam (gradient descent)

model.compile(loss='mape', optimizer=optim, metrics=['mape'])

# fit model with input and output (below line for auto validation
# history = model.fit(X, Y, epochs=1000, batch_size=b_size)

# below line for manual validation
history = model.fit(X_Tr, Y_Tr, validation_data=(X_val, Y_val), epochs=1000, batch_size=b_size)

min_error = min(history.history['mean_absolute_percentage_error'])
min_val_error = min(history.history['val_mean_absolute_percentage_error'])
# save model
save_dir = 'D:\\Derek\\Matlab\\gait_study\\algorithm\\train\\temp\\'
model_name = save_dir + optim + "_" + str(int(min_val_error)) + "_" + str(num_neurons_1st) + "_" + \
             str(num_neurons_2nd) + "_" + str(b_size) + "_" + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ".h5"
model.save(model_name)
print("train complete, minimal mape: %.2f%%" % min_error)
print("minimal validation mape: %.2f%%" % min_val_error)
pyplot.plot(history.history['val_mean_absolute_percentage_error'])
pyplot.show()
