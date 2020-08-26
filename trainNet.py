import csv 
import math

import sys

import numpy as np 
import pandas as pd 
import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import glob
import seaborn as sns
import re
import os
import io 
import pandasql as ps
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import mxnet as mx 
from mxnet import nd, autograd, gluon
from mxnet.gluon.data import DataLoader 
from mxnet.gluon.data.vision import transforms
from mxnet.gluon import nn 




def construct_ff_net():
	ff_net = gluon.nn.HybridSequential()
	with ff_net.name_scope():
		ff_net.add(nn.Flatten())
		ff_net.add(nn.Dense(units=300, activation = 'relu'))
		ff_net.add(nn.Dense(units=200, activation = 'relu'))
		ff_net.add(nn.Flatten())
		ff_net.add(nn.Dense(units=50, activation = 'relu'))
		ff_net.add(nn.Dense(units=10, activation = 'relu'))
		ff_net.add(nn.Dense(units=3, activation = 'relu'))
		ff_net.add(nn.Dense(units=1))
	return ff_net


data_pd = pd.read_csv("fullData.csv")



log_columns = ['r_0','r_1','r_2','r_3','r_4','r_5','r_6','r_7','r_8','r_9','r_10','r_11','r_12','r_13','r_14','r_15','r_16','r_17','r_18','r_19','r_20','r_21','r_22','r_23','r_24','r_25','r_26','r_27','r_28','r_29','r_30','y_0','y_1','y_2','y_3','y_4','y_5','y_6','y_7','y_8','y_9','y_10','y_11','y_12','y_13','y_14','y_15','y_16','y_17','y_18','y_19','y_20','y_21','y_22','y_23','y_24','y_25','y_26','y_27','y_28','y_29','y_30']
	
for col in log_columns:

	data_pd[col] = np.log(data_pd[col] + 1)


label = data_pd['label']
temp_data = data_pd.drop(['label'], axis = 1)
features = temp_data 


x_train,x_test,y_train,y_test = train_test_split(features, label, test_size=0.20, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


pca = PCA(n_components = 62)
pca.fit(x_train)
np.set_printoptions(suppress=True)

print("Explained variance ratio below")

print(pca.explained_variance_ratio_)

plt.plot(np.arange(62), pca.explained_variance_ratio_)

# apply PCA transformation
pca = PCA(n_components = 9)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)



print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

x_train = nd.array(x_train)
x_test = nd.array(x_test)
y_train = nd.array(y_train)
y_test = nd.array(y_test)


# start constructing logistic regression mxnet model 
net = gluon.nn.Dense(1)

ctx = mx.cpu() 



batch_size = 32
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(x_train,y_train), batch_size=32, shuffle=True)

test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(x_test,y_test), batch_size=1, shuffle=True)

criterion = gluon.loss.SigmoidBinaryCrossEntropyLoss()

net.initialize(mx.init.Normal(sigma=1.), ctx=ctx)

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.004})

epochs = 10
num_examples = len(x_train)

for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = criterion(output, label)
        loss.backward()
        trainer.step(batch_size)
        cumulative_loss += nd.sum(loss).asscalar()
    print("Epoch %s, loss: %s" % (e, cumulative_loss ))

num_total = len(x_test)
num_correct = 0
for i, (data, label) in enumerate(test_data):
    data = data.as_in_context(ctx)
    label = label.as_in_context(ctx)
    output = net(data)
    prediction = (nd.sign(output) + 1) / 2
    num_correct += nd.sum(prediction == label)
print("Accuracy: %0.3f (%s/%s)" % (num_correct.asscalar()/num_total, num_correct.asscalar(), num_total))


##################################################################################################
# start of feed foward NN 
ff_net = construct_ff_net()

ff_net.hybridize()

ff_net.initialize(init=mx.init.Xavier(), ctx=ctx)

ff_trainer = gluon.Trainer(ff_net.collect_params(), 'sgd', {'learning_rate': 0.04})


epochs = 10
num_examples = len(x_train)

for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = ff_net(data)
            loss = criterion(output, label)
        loss.backward()
        ff_trainer.step(32)
        cumulative_loss += nd.sum(loss).asscalar()
    print("Epoch %s, loss: %s" % (e, cumulative_loss ))

num_total = len(x_test)
num_correct = 0
for i, (data, label) in enumerate(test_data):
    data = data.as_in_context(ctx)
    label = label.as_in_context(ctx)
    output = ff_net(data)
    prediction = (nd.sign(output) + 1) / 2
    num_correct += nd.sum(prediction == label)

print("Accuracy: %0.3f (%s/%s)" % (num_correct.asscalar()/num_total, num_correct.asscalar(), num_total))


# export file 
ff_net.export("model_ff_net", epoch=10)











