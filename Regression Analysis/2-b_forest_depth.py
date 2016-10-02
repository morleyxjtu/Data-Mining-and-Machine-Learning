'''
Explore the influence of tree depth on the performance in RMSE
Created Jan 2016
@Author: Muchen Xu

'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
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
        data['Size of Backup (GB)'] = data['Size of Backup (GB)'] * 1E9
        data['Size of Backup (GB)'] = data['Size of Backup (GB)'].astype(int)
        return data

train = convert(train)

train['is_train'] = np.random.uniform(0, 1, len(train)) <= .9
train, validate = train[train['is_train']==True], train[train['is_train']==False]

y_train = train['Size of Backup (GB)']
x_train = train.drop('Size of Backup (GB)', 1)
x_train = x_train.drop('is_train',1).as_matrix()

y_validate = validate['Size of Backup (GB)']
x_validate = validate.drop('Size of Backup (GB)', 1)
x_validate = x_validate.drop('is_train',1).as_matrix()

for i in range(39):
	depth=i+1
	rf = RandomForestRegressor(n_estimators=20, max_features = 6, max_depth = depth)
	rf.fit(x_train, y_train)
	disbursed = rf.predict(x_validate)
	sum1=0
	diff=0
	for i in range(len(disbursed.tolist())):
		diff=(disbursed.tolist()[i]-y_validate.tolist()[i])/1E9
		diff_sqr=diff*diff
		sum1=sum1+diff_sqr
	msqr=math.sqrt(sum1/len(disbursed.tolist()))
	print msqr
#	np.savetxt("foo1.csv", disbursed, delimiter=",")
#	np.savetxt("foo2.csv", y_validate, delimiter=",")

