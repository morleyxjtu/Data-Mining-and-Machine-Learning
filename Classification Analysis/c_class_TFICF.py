
# coding: utf-8

# In[13]:

from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups()

categories = newsgroups_train.target_names
data = []
for x in range (0, len(categories)):
    c = []
    c.append(categories[x])
    newsgroups_train_temp = fetch_20newsgroups(categories = c)
    #print help(twenty_train)
    a = newsgroups_train_temp.data[0]
    for x in range(1, len(newsgroups_train_temp.data)):
        a = a + newsgroups_train_temp.data[x]
    data.append(a)


# In[14]:

from sklearn.feature_extraction.text import CountVectorizer

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk

stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

import string

def tokenize(text):
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

from sklearn.feature_extraction import text
stop_words = text.ENGLISH_STOP_WORDS


# In[11]:

vect = CountVectorizer(tokenizer=tokenize, stop_words=stop_words,analyzer='word')


# In[20]:

result = vect.fit_transform(data)


# In[31]:

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(result)


# In[47]:

lst = X_train_tfidf.toarray()[3, :] #comp.sys.ibm.pc.hardware
from heapq import nlargest
a = nlargest(10, enumerate(lst), key=lambda x: x[1])
word = []
for x in range(0, len(a)):
    word.append(vect.vocabulary_.keys()[a[x][0]])
print word


# In[48]:

lst = X_train_tfidf.toarray()[4, :] #comp.sys.ibm.pc.hardware
from heapq import nlargest
a = nlargest(10, enumerate(lst), key=lambda x: x[1])
word = []
for x in range(0, len(a)):
    word.append(vect.vocabulary_.keys()[a[x][0]])
print word


# In[49]:

lst = X_train_tfidf.toarray()[6, :] #comp.sys.ibm.pc.hardware
from heapq import nlargest
a = nlargest(10, enumerate(lst), key=lambda x: x[1])
word = []
for x in range(0, len(a)):
    word.append(vect.vocabulary_.keys()[a[x][0]])
print word


# In[50]:

lst = X_train_tfidf.toarray()[15, :] #comp.sys.ibm.pc.hardware
from heapq import nlargest
a = nlargest(10, enumerate(lst), key=lambda x: x[1])
word = []
for x in range(0, len(a)):
    word.append(vect.vocabulary_.keys()[a[x][0]])
print word


# In[ ]:



