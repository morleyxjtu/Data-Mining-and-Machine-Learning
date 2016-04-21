
# coding: utf-8

# In[7]:

import json
import time
import json
import datetime as datetime
import time
import re
POSITIVE = ["*O", "*-*", "*O*", "*o*", "* *",
            ":P", ":D", ":d", ":p",
            ";P", ";D", ";d", ";p",
            ":-)", ";-)", ":=)", ";=)",
            ":<)", ":>)", ";>)", ";=)",
            "=}", ":)", "(:;)",
            "(;", ":}", "{:", ";}",
            "{;:]",
            "[;", ":')", ";')", ":-3",
            "{;", ":]",
            ";-3", ":-x", ";-x", ":-X",
            ";-X", ":-}", ";-=}", ":-]",
            ";-]", ":-.)",
            "^_^", "^-^"]

NEGATIVE = [":(", ";(", ":'(",
            ":-(", ";-(", ":=(", ";=(",
            "=(", "={", "):", ");",
            ")':", ")';", ")=", "}=",
            ";-{{", ";-{", ":-{{", ":-{",
            ":-(", ";-(",
            ":,)", ":'{",
            "[:", ";]"
            ]
f = open('test_data/sample1_period1_1.txt')
line = f.readline()

date1 = datetime.datetime(2015,02,01, 18,0,0)
keyTime1 = int(time.mktime(date1.timetuple()))
date2 = datetime.datetime(2015,02,02, 0,0,0)
keyTime2 = int(time.mktime(date2.timetuple()))
print keyTime1
print keyTime2

tweet_time = []
while len(line)!=0:
    tweet = json.loads(line)
    if tweet['firstpost_date'] >= keyTime1:
        if tweet['firstpost_date'] > keyTime2:
            break
    tweet_time.append(tweet['firstpost_date'])
    line = f.readline()
f.close()


# In[8]:

initial = tweet_time[0]
last = tweet_time[-1]
print initial
print last


# In[5]:

g = open('test_data/sample1_period1_1.txt')
line = g.readline()

num_positive = [0]* ((last-initial)/360+1)
num_negative = [0]* ((last-initial)/360+1)
emotion = [0]* ((last-initial)/360+1)

while len(line)!=0:
    tweet = json.loads(line)
    if tweet['firstpost_date'] >= keyTime1:
        if tweet['firstpost_date'] > keyTime2:
            break
    n = (tweet['firstpost_date'] - initial)/360
    for happy in POSITIVE:
        if happy in tweet['tweet']['text']:
            num_positive[n] += 1
    for sad in NEGATIVE:
        if sad in tweet['tweet']['text']:
            num_negative[n] -= 1
    emotion[n] = num_positive[n] + num_negative[n]
    line = g.readline()


# In[6]:

import matplotlib.pyplot as plt

plt.bar(range(0,(last-initial)/360+1), num_positive, 1, color = 'b')
plt.bar(range(0,(last-initial)/360+1), num_negative, 1, color = 'r')
plt.show()


# In[ ]:



