'''
Use soft support vector machine (SVM) to seperate the documents into Computer Technology vs Recreational ativity
Created February 2016
@Author: Muchen Xu
'''
##
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
##

#fetch data in two categories
categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
computer = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)
categories = [ 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
recreation = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)

#data and target vector
data = np.concatenate((computer.data, recreation.data), axis = 0)
target = np.concatenate((np.zeros(len(computer.target)), np.zeros(len(recreation.target))+1), axis = 0)

#funtion to remvoe stemming and puctuations
# based on http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
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

#define pipeline for tokenizing, feature extraction, feature selection, and softSVC
parameters = [ 0.001,0.01,0.1, 1, 10, 100, 1000]
for x in range (0, 7):
    text_clf = Pipeline([('vect', CountVectorizer(tokenizer=tokenize, stop_words=stop_words,analyzer='word')),
                     ('tfidf', TfidfTransformer()),
                          ('dimensionality_reduction',TruncatedSVD(n_components=50, random_state=42)),
                       ('clf', SGDClassifier(alpha = parameters[x]))
    ])
    scores = cross_validation.cross_val_score(text_clf, data, target, cv=5, scoring='f1_weighted')
    print scores

