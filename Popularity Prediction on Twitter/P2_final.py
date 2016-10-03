'''
Linear regresion using 5 features to predict number of tweets
Created Mar 2016
@author: Muchen Xu
'''
##
import json
from datetime import datetime
import time
import statsmodels.api as sm
import numpy as np
##

#read each line in the file and stored it in a JSON string
f = open('test_data/sample1_period1.txt')
line = f.readline()
tweet_time = []
while len(line)!=0:
    tweet = json.loads(line)
    tweet_time.append(tweet['firstpost_date'])
    line = f.readline()
f.close()


initial = tweet_time[0]
last = tweet_time[-1]

g = open('test_data/sample1_period1.txt')
line = g.readline()
length = (last-initial)/3600+1
num_tweet = [0]* (length)
num_retweet = [0]* (length)
user = []
for i in range(0, length):
    user.append(dict())
max_followers = [0]* (length)

while len(line)!=0:
    tweet = json.loads(line)
    n = (tweet['firstpost_date'] - initial)/3600
    num_tweet[n] +=1
    num_retweet[n] += tweet['metrics']["citations"]["total"]
    user[n][tweet['author']['name']] = tweet['author']['followers']
    max_followers[n] = max(max_followers[n], tweet['author']['followers'])
    line = g.readline()

#calculate sum of the number of followers of the user
num_follower = []
for x in user:
    if len(x) == 0:
        num_follower.append(0)
    else:
        follower = x.values()
        num_follower.append(reduce(lambda x, y: x + y, follower))
    
initialHour = datetime.fromtimestamp(initial).hour
timeofDay = []
for j in range(0, (last-initial)/3600+1):
    timeofDay.append(j+initialHour)

Y = np.asarray(num_tweet[1:])
X = []
for n in range (0, len(num_tweet)-1):
    element = []
    element.append(num_tweet[n])
    element.append(num_retweet[n])
    element.append(num_follower[n])
    element.append(max_followers[n])
    element.append(timeofDay[n])
    X.append(element)
        
mod = sm.OLS(Y, X)
res = mod.fit()
print res.summary()


# In[15]:

# from sklearn import linear_model
# regr = linear_model.LinearRegression()
# regr.fit(X, Y)


# In[16]:

# print regr.coef_


# In[18]:

# import numpy as np
# from sklearn import cross_validation

# errSumAll = 0
# for i in range(0, 5):
#     X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2)
#     #print X_train, X_test, y_train, y_test
#     LinearR = linear_model.LinearRegression()
#     LinearR.fit(X_train, y_train)
#     predict =  LinearR.predict(X_test)
#     errSum = 0
#     for x in range(0, len(predict)):
#         errSum += abs(predict[x]- y_test[x])
#     print predict
#     print y_test
#     errSumAll += errSum
# error = errSumAll/5

# print error


# In[ ]:



