'''
Use linear SVM for multiple class classification, one to one class
Created February 2016
@Author: Muchen Xu
'''

from sklearn.datasets import fetch_20newsgroups
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import numpy as np
import string
from sklearn import metrics
from sklearn.feature_extraction import text
from sklearn.svm import SVC

from sklearn.decomposition import TruncatedSVD
# In[18]:

x_train = []
y_train = []
for x in range(0, 4):
    categories = [ 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
    A = [categories[x]]
    del categories[x]
    B = categories
    x_train_A = fetch_20newsgroups(subset='train', categories=A, shuffle=True, random_state=42).data
    print len(x_train_A)
    x_train_B = fetch_20newsgroups(subset='train', categories=B, shuffle=True, random_state=42).data
    x_train.append(np.concatenate((x_train_A, np.random.choice(x_train_B, len(x_train_A))), axis=0)) 
    y_train.append(np.concatenate((np.zeros(len(x_train_A)), np.zeros(len(x_train_A))+1), axis=0)) 


# In[19]:

categories = [ 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
x_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42).data
y_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42).target


# In[20]:

print len(x_train[1])
print len(x_test)
print len(y_train[1])
print len(y_test)


# In[21]:

#define tokenize to filter stem of the word and punctuation
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed
def tokenize(text):
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

#obtain stop words
stop_words = text.ENGLISH_STOP_WORDS

#define pipeline

text_clf = Pipeline([('vect', CountVectorizer(tokenizer=tokenize, stop_words=stop_words,analyzer='word')),
                     ('tfidf', TfidfTransformer()),
                     ('dimensionality_reduction',TruncatedSVD(n_components=50, random_state=42)),
                     ('clf', SVC(kernel="linear")),
])


# In[22]:

#obtain predicted value for 6 classifiers
clf1 = text_clf.fit(x_train[0], y_train[0])
pred1 = clf1.predict(x_test)
clf2 = text_clf.fit(x_train[1], y_train[1])
pred2 = clf2.predict(x_test)
clf3 = text_clf.fit(x_train[2], y_train[2])
pred3 = clf3.predict(x_test)
clf4 = text_clf.fit(x_train[3], y_train[3])
pred4 = clf4.predict(x_test)


# In[25]:

#count the vote of each class and store in result
result = np.full((4, len(pred1)), 0, dtype=np.int)
for x in range (0, len(pred1)):
    if pred1[x]==0:
        result[0][x] = result[0][x]+1
for x in range (0, len(pred2)):
    if pred2[x]==0:
        result[1][x] = result[0][x]+1
for x in range (0, len(pred3)):
    if pred3[x]==0:
        result[2][x] = result[0][x]+1
for x in range (0, len(pred4)):
    if pred4[x]==0:
        result[3][x] = result[1][x]+1     
print result


# In[26]:

#see the max vote for each document
prediction = np.zeros(len(result[0]))
for x in range(0, len(pred1)):
    lst = [result[0][x], result[1][x], result[2][x], result[3][x]]
    m = max(lst)
    index = [i for i, j in enumerate(lst) if j == m]
    prediction[x] = index[0]
print prediction


# In[27]:

print np.mean(prediction == y_test)


# In[29]:

names = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
print(metrics.classification_report(y_test, prediction,
    target_names = names))

print metrics.confusion_matrix(y_test, prediction)


# In[36]:

fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.show()


# In[ ]:



