'''
Apply latent semantic indexing (LSI) to TFxIDF data to reduce the dimensionality of the vector
Created February 2016
@Author: Muchen Xu
'''
##
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.datasets import fetch_20newsgroups
import string
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
##
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

##removing stemming and generate tfidf matrix
twenty_train = fetch_20newsgroups(subset = 'train', shuffle = True, random_state = 42);
stop_words = text.ENGLISH_STOP_WORDS  # remove stopwords
vect = CountVectorizer(tokenizer=tokenize, stop_words=stop_words,analyzer='word')
Y=vect.fit_transform(twenty_train.data)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(Y)
Z=X_train_tfidf.toarray()

#reduce the dimension of TFxIDF vector
svd = TruncatedSVD(n_components=50, random_state=42)
dimensionality_reduction=svd.fit_transform(Z)
print dimensionality_reduction