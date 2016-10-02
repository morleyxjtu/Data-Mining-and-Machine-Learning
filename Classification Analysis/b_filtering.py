'''
Tokenize document, excluding the stop words, punctuatons and stemming
Then create the TFxIDF vector representations
Created February 2016
@Author: Muchen Xu
'''

##
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
import string
from sklearn.feature_extraction import text
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

#import data
categories = ['comp.graphics', 'comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
twenty_train = fetch_20newsgroups(categories=categories, shuffle=True, random_state=42)


#CountVecterizer
stop_words = text.ENGLISH_STOP_WORDS  # remove stopwords
vect = CountVectorizer(tokenizer=tokenize, stop_words=stop_words,analyzer='word')
result =  vect.fit_transform(twenty_train.data)

#TFIDF
tfidf_transformer = TfidfTransformer()
text_tfidf = tfidf_transformer.fit_transform(result)





