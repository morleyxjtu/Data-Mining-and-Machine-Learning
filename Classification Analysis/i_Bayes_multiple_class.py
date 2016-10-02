'''
Use soft naïve Bayes algorithm for multiple class classification
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
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
import numpy as np
import string
from sklearn import metrics
from sklearn.feature_extraction import text
from sklearn.decomposition import TruncatedSVD
##

#fetch all the data including training data and test data from both categories
categories = [ 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

#training date and testing data
x_train = train.data
y_train = train.target
x_test = test.data
y_test = test.target

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

#define pipeline for tokenizing, feature extraction, feature selection, and naïve Bayes algorithm
text_clf = Pipeline([('vect', CountVectorizer(tokenizer=tokenize, stop_words=stop_words,analyzer='word')),
                     ('tfidf', TfidfTransformer()),
                      ('dimensionality_reduction',TruncatedSVD(n_components=50, random_state=42)),
                     ('clf', GaussianNB()),
])

text_clf = text_clf.fit(x_train, y_train)

#test data validation
predicted = text_clf.predict(x_test)
print np.mean(predicted == y_test)

#print the statistic summary and confusion matrix
names = ['comp.sys.ibm.pc.hardware' , 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
print(metrics.classification_report(y_test, predicted,
    target_names = names))

print metrics.confusion_matrix(y_test, predicted)

# fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted)
# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.show()



