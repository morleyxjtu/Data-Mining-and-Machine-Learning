from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups()
name = newsgroups_train.target_names

number = []
for x in range(0, len(name)):
	categories = []
	categories.append(name[x]) 
	graph = fetch_20newsgroups(subset = 'train', categories = categories, shuffle = True, random_state = 42)
	number.append(graph.filenames.shape[0]) 

import numpy as np
import matplotlib.pyplot as plt

#alphab = name
#frequencies = number

#pos = np.arange(len(alphab))
#width = 1.0     # gives histogram aspect to the bar diagram

#ax = plt.axes()
#ax.set_xticks(pos + (width / 2))
#ax.set_xticklabels(alphab)

#plt.bar(pos, frequencies, width, color='r')
#plt.show()

fig, ax = plt.subplots()

index = np.arange(20)
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index,number , bar_width, alpha=opacity,color='b',yerr=0,error_kw=error_config)


plt.xlabel('Topics')
plt.ylabel('number of documents')
plt.title('number of documents per topic')
plt.xticks(index + bar_width, ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'))
plt.legend()

plt.tight_layout()
plt.show()

print number[1]+number[2]+number[3]+number[4]
print number[7]+number[8]+number[9]+number[10]