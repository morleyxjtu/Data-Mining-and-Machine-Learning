'''
Linear regresion using extra features to predict number of tweets
Used Random Forrest Regression model
Used cross_validation
For tweets all time
Extra features to explore: co-occurrence times of other hashtags; author count; mention count; Special signals; Emotion count;

Created Mar 2016
@author: Muchen Xu
'''

import json
from datetime import datetime
import time
import json
f = open('test_data/sample1_period1.txt')
line = f.readline()

tweet_time = []
while len(line)!=0:
    tweet = json.loads(line)
    tweet_time.append(tweet['firstpost_date'])
    line = f.readline()
f.close()


# In[49]:

initial = tweet_time[0]
last = tweet_time[-1]


# In[212]:

g = open('test_data/sample1_period1.txt')
line = g.readline()

length = (last-initial)/3600+1
num_tweet = [0]* (length)
user = []
for i in range(0, length):
    user.append(dict())
mention_num = [0]*(length)
hashtag_num = [0]*(length)
url_exist = [0]*(length)
num_retweet = [0]* (length)
max_followers = [0]* (length)

while len(line)!=0:
    tweet = json.loads(line)
    n = (tweet['firstpost_date'] - initial)/3600
    num_tweet[n] +=1
    mention_num[n] += len(tweet['tweet']['entities']['user_mentions'])
    hashtag_num[n] += len(tweet['tweet']['entities']['hashtags'])
    if (tweet['tweet']['entities']['urls']):
        url_exist[n] += 1
        
    num_retweet[n] += tweet['metrics']["citations"]["total"]
    user[n][tweet['author']['name']] = tweet['author']['followers']
    max_followers[n] = max(max_followers[n], tweet['author']['followers'])
    line = g.readline()
        
#number of users in each hour
user_num = []
for m in user:
    user_num.append(len(m))
    
url_ratio = []
for x in range(0, (last-initial)/3600+1):
    if num_tweet[x] == 0:
        url_ratio.append(0)
    else:
        url_ratio.append(url_exist[x]/num_tweet[x])
    
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


# In[167]:

import numpy as np
Y = np.asarray(num_tweet[1:])

X = []
for n in range (0, len(num_tweet)-1):
    element = []
    element.append(user_num[n])
    element.append(mention_num[n])
    element.append(hashtag_num[n])
    element.append(url_ratio[n])
    element.append(num_tweet[n])
    element.append(num_retweet[n])
    element.append(num_follower[n])
    element.append(max_followers[n])
    element.append(timeofDay[n])
    X.append(element)
    
X = np.asarray(X)

#print X


# In[178]:

# from sklearn.ensemble import RandomForestRegressor
# rf = RandomForestRegressor(n_estimators=20, max_features = 6, max_depth = 4)
# rf.fit(X, Y)


# In[ ]:

# importances = forest.feature_importances_


# In[168]:

# from sklearn import linear_model
# regr = linear_model.LinearRegression()
# regr.fit(X, Y)
# regr.coef_


# In[169]:

# import numpy as np
# import statsmodels.api as sm
# X = sm.add_constant(X)
# mod = sm.OLS(Y, X)
# res = mod.fit()
# print res.summary()


# In[211]:

import numpy as np
from sklearn import cross_validation

errSumAll = 0
for i in range(0, 10):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.1)
    #print X_train, X_test, y_train, y_test
    rf = RandomForestRegressor(n_estimators=40, max_features = 9, max_depth = 10)
    rf.fit(X_train, y_train)
    predict =  rf.predict(X_test)
    errSum = 0
    for x in range(0, len(predict)):
        errSum += abs(predict[x]- y_test[x])
    print predict
    print y_test
    print errSum/len(predict)
    errSumAll += errSum/len(predict)

error = errSumAll/10

print error
g.close()


# In[ ]:




# In[ ]:




# In[ ]:



