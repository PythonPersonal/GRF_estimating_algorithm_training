from findAllIndicesInList import all_indices
import pandas

data = pandas.read_csv("D:\\Derek\\Matlab\\gait_study\\algorithm\\train\\data\\SubFeatures_stage2.csv",delimiter=',',header=0)
data_set = data.values
print(all_indices(4, list(data_set[:,0])))
