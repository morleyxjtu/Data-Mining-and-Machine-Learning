'''
Use a neural network regression model and explore the influence of major parameters
Created Jan 2016
@Author: Muchen Xu

'''

import pandas as pd
import numpy as np
from sklearn import preprocessing
import neurolab as nl
import math
import os
number = preprocessing.LabelEncoder()
path=os.getcwd()
train=pd.read_csv(path+'/network_backup_dataset.csv')

def convert(data):
	number = preprocessing.LabelEncoder()
	data['Day of Week'] = number.fit_transform(data['Day of Week'])
	data['Work-Flow-ID'] = number.fit_transform(data['Work-Flow-ID'])
	data['File Name'] = number.fit_transform(data['File Name'])
#	data['Size of Backup (GB)'] = data.Size of Backup (GB) * 1E9
#	data['Size of Backup (GB)'] = data['Size of Backup (GB)'].astype(int)
	return data

train = convert(train)
print train
train['is_train'] = np.random.uniform(0, 1, len(train)) <= .90
train, validate = train[train['is_train']==True], train[train['is_train']==False]

y_train = train['Size of Backup (GB)'].as_matrix()
x_train = train.drop('Size of Backup (GB)', 1)
x_train = x_train.drop('is_train',1).as_matrix()
size = len(x_train)
x_train = x_train.reshape(size, 6)
y_train = y_train.reshape(size, 1)


y_validate = validate['Size of Backup (GB)'].as_matrix()
x_validate = validate.drop('Size of Backup (GB)', 1)
x_validate = x_validate.drop('is_train',1).as_matrix()
size = len(x_validate)
x_validate = x_validate.reshape(size, 6)
y_validate = y_validate.reshape(size, 1)

# Create network with 2 layers and random initialized
net = nl.net.newff([[1, 15], [0, 6], [1, 21], [0, 4], [0, 29],[0, 4]], [5,3,1]) #[3,1] could be changed, e.g., [5, 1]
#net = nl.net.newff([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1],[0, 1]], [12, 1])
# Train network
msqr_array=[]
for j in range(5):
	epoch_time=20*j
	net.trainf=nl.train.train_bfgs
	error = net.train(x_train, y_train, epochs=epoch_time, show=1, goal=0.00002) #all these parameters could also be changed
# Simulate network
	disbursed = net.sim(x_validate)
	sum1=0
	for i in range(len(disbursed.tolist())):
		diff=(disbursed.tolist()[i][0]-y_validate.tolist()[i][0])/1E9
        	diff_sqr=diff*diff
        	sum1=sum1+diff_sqr
	msqr=math.sqrt(sum1/len(disbursed.tolist()))
	print msqr
	msqr_array.append(msqr)
	print msqr_array
print msqr_array
#print x_validate
#np.savetxt("foo3.csv", disbursed, delimiter=",")
#np.savetxt("foo4.csv", y_validate, delimiter=",")

