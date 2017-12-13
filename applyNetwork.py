from keras.models import load_model
import pandas
from findAllIndicesInList import all_indices
import numpy
import scipy.io
from matplotlib import pyplot as plt

model_dir = 'D:\\Derek\\Matlab\\gait_study\\algorithm\\train\\temp\\adam_6_30_30_32_2017-12-13-16-56-07.h5'
model = load_model(model_dir)
data = pandas.read_csv("D:\\Derek\\Matlab\\gait_study\\algorithm\\train\\data\\SubFeatures_stage2.csv",
                       delimiter=',', header=0)
data_set = data.values
SubIdxForValidate = [4, 5, 6, 11, 20]
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
# predict
Y_predict = model.predict(X_val)
x_axis = numpy.linspace(0, len(Y_predict)-1, len(Y_predict))
plt.scatter(x_axis, Y_predict, color='blue', label='predicted value', marker='x', s=10)
plt.scatter(x_axis, Y_val, color='red', label='expected value', marker='o', s=10)
plt.title('Results: Adam 15/15')
plt.legend(loc='best')
plt.show()
# save for external use
weights = model.get_weights()
scipy.io.savemat(
    'D:\\Derek\\Matlab\\gait_study\\algorithm\\train\\temp\\export\\adam_6_30_30_32_2017-12-13-16-56-07.mat',
    mdict={'net': weights})