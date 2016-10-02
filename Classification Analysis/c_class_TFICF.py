'''
Find the 10 most significant terms in certain documents using TFxICF
Created February 2016
@Author: Muchen Xu
'''

##
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk
import string
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
from heapq import nlargest
##

#funtion to remvoe stemming and puctuations
# based on http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

stemmer = PorterStemmer()
def tokenize(text):
    text = "".join([ch for ch in text if ch not in string.punctuation]) #remove punctuation
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer) #remove stemming
    return stems

#combine all the douments in one category into one document
newsgroups_train = fetch_20newsgroups()
categories = newsgroups_train.target_names
data = []
for x in range (0, len(categories)):
    c = []
    c.append(categories[x])
    newsgroups_train_temp = fetch_20newsgroups(categories = c) #fetch all one category
    a = newsgroups_train_temp.data[0] #fetch the data part of that category
    #combine all data in the category
    for x in range(1, len(newsgroups_train_temp.data)):
        a = a + newsgroups_train_temp.data[x]
    data.append(a)

##removing stemming and generate tfidf matrix
stop_words = text.ENGLISH_STOP_WORDS
vect = CountVectorizer(tokenizer=tokenize, stop_words=stop_words,analyzer='word')
result = vect.fit_transform(data)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(result)


# 'comp.sys.ibm.pc.hardware' category
lst = X_train_tfidf.toarray()[3, :] 
a = nlargest(10, enumerate(lst), key=lambda x: x[1])
word = []
for x in range(0, len(a)):
    word.append(vect.vocabulary_.keys()[a[x][0]])
print word

#'comp.sys.ibm.pc.hardware' category
lst = X_train_tfidf.toarray()[4, :] 
a = nlargest(10, enumerate(lst), key=lambda x: x[1])
word = []
for x in range(0, len(a)):
    word.append(vect.vocabulary_.keys()[a[x][0]])
print word

#'comp.sys.ibm.pc.hardware' category
lst = X_train_tfidf.toarray()[6, :] 
a = nlargest(10, enumerate(lst), key=lambda x: x[1])
word = []
for x in range(0, len(a)):
    word.append(vect.vocabulary_.keys()[a[x][0]])
print word


#'comp.sys.ibm.pc.hardware'category
lst = X_train_tfidf.toarray()[15, :] 
a = nlargest(10, enumerate(lst), key=lambda x: x[1])
word = []
for x in range(0, len(a)):
    word.append(vect.vocabulary_.keys()[a[x][0]])
print word



