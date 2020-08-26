import csv 
import math

import numpy as np 
import pandas as pd 
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


import mxnet as mx 
import warnings
from mxnet import nd, autograd, gluon
from mxnet.gluon.data import DataLoader 
from mxnet.gluon.data.vision import transforms
from mxnet.gluon import nn 



# THE FILE PATH IS THE ONLY THING YOU NEED TO EDIT
with open('bpmData/test.bpm.100.y.csv', 'r') as csvfile:
	reader = csv.reader(csvfile)
	my_list = list(reader)
	my_list.remove(my_list[0])
	my_list.remove(my_list[-1])
	my_list.remove(my_list[-1])

	s,y = map(list,zip(*my_list))

	cutoff = 31 
	if (len(s) < 31):
		cutoff = len(s)

	# takes data in two column format and makes it into one list
	temparr1 = [None] * cutoff
	temparr2 = [None] * cutoff

	for num1 in range(0,cutoff,1):
		temparr1[num1] = s[num1]
		temparr2[num1] = y[num1]

	temparr1 = np.array(temparr1, dtype = float)
	temparr2 = np.array(temparr2, dtype = float)

	entry1 = temparr1.transpose()
	entry2 = temparr2.transpose()
	entry = [*entry1, *entry2]


sample_data = entry 


## now manipulating data

data_pd = pd.read_csv("fullData.csv")

log_columns = ['r_0','r_1','r_2','r_3','r_4','r_5','r_6','r_7','r_8','r_9','r_10','r_11','r_12','r_13','r_14','r_15','r_16','r_17','r_18','r_19','r_20','r_21','r_22','r_23','r_24','r_25','r_26','r_27','r_28','r_29','r_30','y_0','y_1','y_2','y_3','y_4','y_5','y_6','y_7','y_8','y_9','y_10','y_11','y_12','y_13','y_14','y_15','y_16','y_17','y_18','y_19','y_20','y_21','y_22','y_23','y_24','y_25','y_26','y_27','y_28','y_29','y_30']


# take log to decrease range of data
for col in log_columns:
	data_pd[col] = np.log(data_pd[col] + 1)


# takes single example without label ############################################
features_single = sample_data


# also takes log to decrease range of test data
for i in range(len(features_single)):
	features_single[i] = math.log(features_single[i] + 1)



label = data_pd['label']
temp_data = data_pd.drop(['label'], axis = 1)
features = temp_data


features_single = np.asarray(features_single, dtype=np.float32)
features_single = features_single.reshape(1, -1)



x_train,x_test,y_train,y_test = train_test_split(features, label, test_size=0.20, random_state=42)

# used to scale data again 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
features_single = scaler.transform(features_single)



components = 2 * cutoff
pca = PCA(n_components = components)
pca.fit(x_train)
np.set_printoptions(suppress=True)


# decrease data to only have 9 features
pca = PCA(n_components = 9)
x_train = pca.fit_transform(x_train)
features_single = pca.transform(features_single)


features_single = nd.array(features_single)

# choose cpu to use
ctx = mx.cpu() 

# gets the previously built model from trainNet.py
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	deserialized_net = gluon.nn.SymbolBlock.imports("model_ff_net-symbol.json", ['data'], "model_ff_net-0010.params", ctx=ctx)

net = deserialized_net
output = net(features_single)
prediction = (nd.sign(output) + 1) / 2


if (prediction[0][0] == 0):
	print("This graph is of type BPM")
else:
	print("This graph is of type CORR")





