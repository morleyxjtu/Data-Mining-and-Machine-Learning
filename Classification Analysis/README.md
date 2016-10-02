Goal:

Practice difference methods for classifying textual data. Train different classifiers to classify "20 Newsgroups" data set (~20,000 documents) and evaluate classification performance.

Classification process general description:
       
-- Text processing and feature extraction:
	
        1. Make sure datasets in different classes are balanced
	
        2. Tokenize document, excluding stop words, puctuations, and different stems of a word
	
        3. Create TFxIDF vector representations
	
        4. Apply Latent Semantic Indexing (LSI)to the TFxIDF matrix to reduce its dimension

-- Learning algorithms:
	
        1. Two classes: applied linear SVM machine, soft margin SVM, naïve Bayes algorithm, and logistic regression classifier and compared their performance
	
        2. Multiple clases: applied naïve Bayes algorithm. For SVM, used both one vs one and one vs other stratagies.
