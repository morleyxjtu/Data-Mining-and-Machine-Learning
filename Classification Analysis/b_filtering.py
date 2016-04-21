#part(b)
#take the eight groups that need to be classified

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfTransformer

#import data
#categories = ['alt.atheism','comp.graphics', 'comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','comp.windows.x','misc.forsale','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey','sci.crypt','sci.electronics','sci.med','sci.space','soc.religion.christian','talk.politics.guns','talk.politics.mideast','talk.politics.misc','talk.religion.misc']
categories = ['comp.graphics', 'comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']

twenty_train = fetch_20newsgroups(categories=categories, shuffle=True, random_state=42)
#twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

#filter stop words, stemming, and punctuation
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

import string

def tokenize(text):
    text = "".join([ch for ch in text if ch not in string.punctuation]) #remove punctuation
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer) #remove stemming
    return stems

from sklearn.feature_extraction import text
stop_words = text.ENGLISH_STOP_WORDS  # remove stopwords

#CountVecterizer
vect = CountVectorizer(tokenizer=tokenize, stop_words=stop_words,analyzer='word')
result =  vect.fit_transform(twenty_train.data)
print result.shape

#TFIDF
tfidf_transformer = TfidfTransformer()
text_tfidf = tfidf_transformer.fit_transform(result)
print text_tfidf.shape





