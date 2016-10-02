
'''
Plot the number of documents per topic to make sure they are evenly distributed

Created February 2016
@Author: Muchen Xu
'''
##
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import matplotlib.pyplot as plt
##

##fetch all the news from the 20newsgroups
newsgroups_train = fetch_20newsgroups() #use the data in 20newsgroups as train data
name = newsgroups_train.target_names #extract different categories in the group

##store the number of files of each categories into number[]
number = []
for x in range(0, len(name)):
	categories = [name[x]]
	graph = fetch_20newsgroups(subset = 'train', categories = categories, shuffle = True, random_state = 42)
	number.append(graph.filenames.shape[0]) 

##define the appearance of the histogram
fig, ax = plt.subplots()
index = np.arange(20)
bar_width = 0.35
opacity = 0.4
error_config = {'ecolor': '0.3'}

##define the legend and lable of the histogram
plt.xlabel('Topics')
plt.ylabel('number of documents')
plt.title('number of documents per topic')
plt.xticks(index + bar_width, ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'))

##plot the histogram
plt.bar(index,number , bar_width, alpha=opacity,color='b',yerr=0,error_kw=error_config)
plt.show()