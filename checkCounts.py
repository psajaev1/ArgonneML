import csv 
import numpy as np 
from scipy.optimize import curve_fit
import math

import csv 
import numpy as np 
import sympy as sym 
from scipy.optimize import curve_fit
from mpl_toolkits import mplot3d
from scipy.integrate import trapz
import math
import matplotlib.pyplot as plt 
import random
import matplotlib.pyplot as plt 

import os 

max_len = 31


fileCount = 0
# y data should be easier to analyze, so ignoring all x files
for i in range(0,10000,1):

	with open('/Users/Phillip/ArgonneML/bpmData/test.bpm.{}.y.csv'.format(i), 'r') as csvfile:
			reader = csv.reader(csvfile)
			my_list = list(reader)
			my_list.remove(my_list[-1])
			my_list.remove(my_list[-1])
			s, y = map(list,zip(*my_list))
			length = len(s)
			if length > max_len: 
				fileCount+=1



print("Amount of files over cutoff is: {}".format(fileCount))



fileCount = 0
# y data should be easier to analyze, so ignoring all x files
for i in range(0,10000,1):

	with open('/Users/Phillip/ArgonneML/corrData/test.corr.{}.y.csv'.format(i), 'r') as csvfile:
		reader = csv.reader(csvfile)
		my_list = list(reader)
		my_list.remove(my_list[-1])
		my_list.remove(my_list[-1])
		s, y = map(list,zip(*my_list))
		length = len(s)
		if length > max_len: 
			fileCount+=1



print("Amount of files over cutoff is: {}".format(fileCount))




