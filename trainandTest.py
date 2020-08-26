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

#see how logistic regression does
logreg = LogisticRegression()
logreg.fit(x_train,y_train)

y_pred = logreg.predict(x_test)

print(y_pred)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))

confusion_matrix_log = confusion_matrix(y_test, y_pred)
print(confusion_matrix_log)

#### RANDOM FOREST REGRESSION NOW

estimator = RandomForestRegressor()
parameters = {'n_estimators':[1],'max_depth':[40,60,80,100,120]}  
clf = GridSearchCV(estimator, parameters, n_jobs = -1)
clf.fit(x_train,y_train)

print(clf.best_params_)
print(clf.best_score_)

rando = RandomForestRegressor(n_estimators=1, max_depth=40, n_jobs=-1)
rando = rando.fit(x_train,y_train)
y_pred_forest = rando.predict(x_test)

y_pred_forest = y_pred_forest.tolist()
y_test = y_test.tolist()
print("Accuracy of random forest classifier on test set: {:.0%}".format(accuracy_score(y_test,y_pred_forest)))


confusion_matrix_forest = confusion_matrix(y_test,y_pred_forest)
print(confusion_matrix_forest)