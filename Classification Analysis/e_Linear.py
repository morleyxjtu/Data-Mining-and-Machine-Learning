'''
Use linear support vector machine (SVM) to seperate the documents into Computer Technology vs Recreational ativity
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
from sklearn.pipeline import Pipeline
import numpy as np
import string
from sklearn import metrics
from sklearn.feature_extraction import text
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import TruncatedSVD
##

#fetch all the data including training data and test data from both categories
categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
computer_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
computer_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
categories = [ 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
recreation_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
recreation_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

#set the target value for computer (0) and recreation (1)
comp_target = np.zeros(len(computer_train.target))
recr_target = np.zeros(len(recreation_train.target))+1
comp_target2 = np.zeros(len(computer_test.target))
recr_target2 = np.zeros(len(recreation_test.target))+1

#concatenate data from both categories together, where x is the document data, y is the category data
x_train = np.concatenate((computer_train.data, recreation_train.data), axis=0)
y_train = np.concatenate((comp_target, recr_target), axis=0)
x_test = np.concatenate((computer_test.data, recreation_test.data), axis=0)
y_test = np.concatenate((comp_target2, recr_target2), axis=0)

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

#define pipeline for tokenizing, feature extraction, feature selection, and linear SVC
text_clf = Pipeline([('vect', CountVectorizer(tokenizer=tokenize, stop_words=stop_words,analyzer='word')),
                     ('tfidf', TfidfTransformer()),
                     ('dimensionality_reduction',TruncatedSVD(n_components=50, random_state=42)),
                     ('clf', LinearSVC())
])
text_clf = text_clf.fit(x_train, y_train)

#Use the classifier to predict
predicted = text_clf.predict(x_test)
print np.mean(predicted == y_test)

#print the statistic summary and confusion matrix
names = ['Computer Tech', 'Recreation']
print(metrics.classification_report(y_test, predicted,
    target_names = names))

conf=metrics.confusion_matrix(y_test, predicted)
print conf

#plot the receiver operating characteristic (ROC) curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted)
print fpr
print tpr
import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.show()



