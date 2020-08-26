import csv 
import numpy as np 
import math

import numpy as np 
import sys


typegraph1 = []
typegraph1.append(0)
typegraph2 = []
typegraph2.append(1)

cutoff = 31

temparr1 = [None] * cutoff
temparr2 = [None] * cutoff

column_names = []
a = 0
b = 0
for i in range(0,63,1):
	if i < 31:
		column_names.append('r_{}'.format(a))
		a+=1
	elif i < 62:
		column_names.append('y_{}'.format(b))
		b+=1


with open('fullData.csv', 'w') as f:
	f.write('r_0,r_1,r_2,r_3,r_4,r_5,r_6,r_7,r_8,r_9,r_10,r_11,r_12,r_13,r_14,r_15,r_16,r_17,r_18,r_19,r_20,r_21,r_22,r_23,r_24,r_25,r_26,r_27,r_28,r_29,r_30,y_0,y_1,y_2,y_3,y_4,y_5,y_6,y_7,y_8,y_9,y_10,y_11,y_12,y_13,y_14,y_15,y_16,y_17,y_18,y_19,y_20,y_21,y_22,y_23,y_24,y_25,y_26,y_27,y_28,y_29,y_30,label')
	f.write("\n")


count0 = 0
for i in range(0,10000,1):

	with open('/Users/Phillip/ArgonneML/bpmData/test.bpm.{}.y.csv'.format(i), 'r') as csvfile:
		reader = csv.reader(csvfile)
		my_list = list(reader)
		my_list.remove(my_list[0])
		my_list.remove(my_list[-1])
		my_list.remove(my_list[-1])

		r,y = map(list,zip(*my_list))

		if (len(r) < 31):
			continue


		# making it so that only lists with lengths of 31 are inputted
		for num1 in range(0,cutoff,1):
			temparr1[num1] = r[num1]
			temparr2[num1] = y[num1]

		temparr1 = np.array(temparr1, dtype = float)
		temparr2 = np.array(temparr2, dtype = float)



		entry1 = temparr1.transpose()
		entry2 = temparr2.transpose()

		entry = [*entry1,*entry2,*typegraph1]

		length = len(entry)

		with open('fullData.csv', 'a') as f:
			count0 = 0
			for item in entry:
				if count0 == (length-1):
					f.write("{}".format(item))
					f.write("\n")
				else:
					f.write("{},".format(item))
					count0 = count0 + 1




temparr3 = [None] * cutoff
temparr4 = [None] * cutoff

count = 0
for j in range(0,10000,1):

	with open('/Users/Phillip/ArgonneML/corrData/test.corr.{}.y.csv'.format(j), 'r') as csvfile:
		reader = csv.reader(csvfile)
		my_list = list(reader)
		my_list.remove(my_list[0])
		my_list.remove(my_list[-1])
		my_list.remove(my_list[-1])

		r2,y2 = map(list,zip(*my_list))


		if (len(r2) < 31):
			continue

		# making it so that only lists with lengths of 31 are inputted
		for num2 in range(0,cutoff,1):
			temparr3[num2] = r2[num2]
			temparr4[num2] = y2[num2]

		temparr3 = np.array(temparr3, dtype= float)
		temparr4 = np.array(temparr4,dtype = float)

		entry3 = temparr3.transpose()
		entry4 = temparr4.transpose()

		entries = [*entry3, *entry4, *typegraph2]
		entrylen = len(entries)

		with open('fullData.csv','a') as f:
			count = 0
			for item in entries:
				if count == (entrylen-1):
					f.write("{}".format(item))
					f.write("\n")
				else:
					f.write("{},".format(item))
					count = count + 1





	

