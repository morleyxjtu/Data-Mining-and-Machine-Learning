
# coding: utf-8

# In[1]:

from sklearn.datasets import fetch_20newsgroups
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import string
from sklearn import metrics
from sklearn.feature_extraction import text
from sklearn import cross_validation
from sklearn.decomposition import TruncatedSVD

# In[3]:

categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
computer = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)
categories = [ 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
recreation = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)


# In[4]:

data = np.concatenate((computer.data, recreation.data), axis = 0)
target = np.concatenate((np.zeros(len(computer.target)), np.zeros(len(recreation.target))+1), axis = 0)


# In[8]:

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


# In[9]:

#define pipeline



# In[12]:

parameters = [ 0.001,0.01,0.1, 1, 10, 100, 1000]
#parameter=1000


# In[11]:

for x in range (0, 7):
    text_clf = Pipeline([('vect', CountVectorizer(tokenizer=tokenize, stop_words=stop_words,analyzer='word')),
                     ('tfidf', TfidfTransformer()),
                          ('dimensionality_reduction',TruncatedSVD(n_components=50, random_state=42)),
                       ('clf', SGDClassifier(alpha = parameters[x]))
    ])
    scores = cross_validation.cross_val_score(text_clf, data, target, cv=5, scoring='f1_weighted')
    print scores

