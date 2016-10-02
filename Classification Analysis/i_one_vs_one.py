'''
Use linear SVM for multiple class classification, one to multiple class
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



# In[38]:

#training date and testing data
categories = [ 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
x_train = []
y_train = []
for n in range (0, 4):
    for x in range (n+1, 4):
        new = [categories[n], categories[x]]
        print new
        x_train.append(fetch_20newsgroups(subset='train', categories=new, shuffle=True, random_state=42).data)
        y_train.append(fetch_20newsgroups(subset='train', categories=new, shuffle=True, random_state=42).target)
x_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42).data
y_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42).target


# In[44]:

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
                     ('clf', SVC(kernel="linear", )),
])


# In[45]:

#obtain predicted value for 6 classifiers
clf1 = text_clf.fit(x_train[0], y_train[0])
pred1 = clf1.predict(x_test)
clf2 = text_clf.fit(x_train[1], y_train[1])
pred2 = clf2.predict(x_test)
clf3 = text_clf.fit(x_train[2], y_train[2])
pred3 = clf3.predict(x_test)
clf4 = text_clf.fit(x_train[3], y_train[3])
pred4 = clf4.predict(x_test)
clf5 = text_clf.fit(x_train[4], y_train[4])
pred5 = clf5.predict(x_test)
clf6 = text_clf.fit(x_train[5], y_train[5])
pred6 = clf6.predict(x_test)


# In[56]:

#count the vote of each class and store in result
result = np.full((4, len(pred1)), 0, dtype=np.int)
for x in range (0, len(pred1)):
    if pred1[x]==0:
        result[0][x] = result[0][x]+1
    else:
        result[1][x] = result[1][x]+1
for x in range (0, len(pred2)):
    if pred2[x]==0:
        result[0][x] = result[0][x]+1
    else:
        result[2][x] = result[2][x]+1
for x in range (0, len(pred3)):
    if pred3[x]==0:
        result[0][x] = result[0][x]+1
    else:
        result[3][x] = result[3][x]+1
for x in range (0, len(pred4)):
    if pred4[x]==0:
        result[1][x] = result[1][x]+1
    else:
        result[2][x] = result[2][x]+1
for x in range (0, len(pred5)):
    if pred5[x]==0:
        result[1][x] = result[1][x]+1
    else:
        result[3][x] = result[3][x]+1
for x in range (0, len(pred6)):
    if pred6[x]==0:
        result[2][x] = result[2][x]+1
    else:
        result[3][x] = result[3][x]+1
        
print result


# In[57]:

#see the max vote for each document
prediction = np.zeros(len(result[0]))
for x in range(0, len(pred1)):
    lst = [result[0][x], result[1][x], result[2][x], result[3][x]]
    m = max(lst)
    index = [i for i, j in enumerate(lst) if j == m]
    prediction[x] = index[0]
print prediction


# In[58]:

print np.mean(prediction == y_test)


# In[59]:

names = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
print(metrics.classification_report(y_test, prediction,
    target_names = names))

print metrics.confusion_matrix(y_test, prediction)


# In[ ]:



