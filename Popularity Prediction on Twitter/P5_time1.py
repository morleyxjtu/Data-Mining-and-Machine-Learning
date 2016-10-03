'''
Use previous linear regression model for time period #1 to predict number of tweets

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

initial = tweet_time[0]
last = tweet_time[-1]

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


# In[11]:

from sklearn.externals import joblib
gohawk = joblib.load('P4_time1_#gohawks.pkl') 
print "gohawk score ", gohawk.score(X, Y)
predict = gohawk.predict(X)
errSum = 0
for n in range(0, len(predict)):
    errSum += abs(predict[n]- Y[n])
print "gohawk error ", errSum/len(predict)

gopatriots = joblib.load('P4_time1_#gopatriots.pkl') 
print "gopatriots score", gopatriots.score(X, Y)
predict2 = gohawk.predict(X)
errSum2 = 0
for n in range(0, len(predict2)):
    errSum2 += abs(predict2[n]- Y[n])
print "gopatriots error ", errSum2/len(predict2)

nfl = joblib.load('P4_time1_#nfl.pkl') 
print "nfl score", nfl.score(X, Y)
predict3 = gohawk.predict(X)
errSum3 = 0
for n in range(0, len(predict3)):
    errSum3 += abs(predict3[n]- Y[n])
print "nfl error ", errSum3/len(predict3)

patriots = joblib.load('P4_time1_#patriots.pkl') 
print "patriots score ", patriots.score(X, Y)
predict4 = gohawk.predict(X)
errSum4 = 0
for n in range(0, len(predict4)):
    errSum4 += abs(predict4[n]- Y[n])
print "patriots error ", errSum4/len(predict4)

sb49 = joblib.load('P4_time1_#sb49.pkl') 
print "sb49 score", sb49.score(X, Y)
predict5 = gohawk.predict(X)
errSum5 = 0
for n in range(0, len(predict5)):
    errSum5 += abs(predict5[n]- Y[n])
print "sb49 error ", errSum5/len(predict5)

superbowl = joblib.load('P4_time1_#superbowl.pkl') 
print "superbowl score", superbowl.score(X, Y)
predict6 = gohawk.predict(X)
errSum6 = 0
for n in range(0, len(predict6)):
    errSum6 += abs(predict6[n]- Y[n])
print "superbowl error ", errSum6/len(predict6)


# In[ ]:




# In[ ]:



