'''
Explore the influence of tree pattern on the performance in RMSE
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
path=os.getcwd()
number = preprocessing.LabelEncoder()
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
train1, validate = train[train['is_train']==True], train[train['is_train']==False]

y_train = train1['Size of Backup (GB)']
x_train = train1.drop('Size of Backup (GB)', 1)
x_train = x_train.drop('is_train',1).as_matrix()

y_validate = train['Size of Backup (GB)']
x_validate = train.drop('Size of Backup (GB)', 1)
x_validate = x_validate.drop('is_train',1).as_matrix()
rf = RandomForestRegressor(n_estimators=30, max_features = 6, max_depth = 8)
rf.fit(x_train, y_train)
pattern = rf.predict(x_validate)
pattern=pattern*1E-9
np.savetxt("pattern.csv", pattern, delimiter=",")#This is the pattern generated to verify

sum1=0
diff=0
for i in range(len(pattern.tolist())):
	diff=(pattern.tolist()[i]-y_validate.tolist()[i])/1E9
	diff_sqr=diff*diff
	sum1=sum1+diff_sqr
	msqr=math.sqrt(sum1/len(pattern.tolist()))
print msqr
#	np.savetxt("foo1.csv", disbursed, delimiter=",")
#	np.savetxt("foo2.csv", y_validate, delimiter=",")

