'''
Calculate statistics of each hashtag
Created Mar 2016
@author: Muchen Xu
'''
##
import json
import datetime, time
##

#read each line in the file and stored it in a JSON string
f = open('test_data/sample1_period1.txt')
line = f.readline()
tweet = json.loads(line)
tweets = []
while len(line)!=0:
    tweet = json.loads(line)
    tweets.append(tweet)
    line = f.readline()
f.close()

#for each tweet, store the time, retweets, user-follower pair
num_tweets = len(tweets)
max_followers = 0
times = []
retweet = []
d = dict()
for i in range(0, num_tweets):
    tweet = tweets[i]
    tweet_time = tweet['firstpost_date']
    times.append(tweet_time)
    retweet.append(tweet['metrics']["citations"]["total"])
    d[tweet['author']['name']] = tweet['author']['followers']

#convert times to offset hour from initail and save in new_times
initial = min(times)
new_times = []
for x in times:
    new_times.append((x-initial)/3600)

#average number of tweets per hour
numTweetPerHour = [0]*(new_times[-1]-new_times[0]+1)
for x in new_times:
    numTweetPerHour[x] += 1
print reduce(lambda x, y: x + y, numTweetPerHour) / float (len(numTweetPerHour))

#average number of followers
follower_num = d.values()
print reduce(lambda x, y: x + y, follower_num) / float(len(follower_num))

#average number of retweet
print reduce(lambda x, y: x + y, retweet) / float(len(retweet))


# import matplotlib.pyplot as plt
# plt.bar(range(0,new_times[-1]-new_times[0]+1), numTweetPerHour, 1)
# plt.show()




