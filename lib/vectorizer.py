from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np

class MeanEmbeddingVectorizer(object):
    def __init__(self, w2v):
        self.word2vec = w2v # dict structure. keys are words, values are embedding
        self.dim = list(w2v.items())[0][1].shape[0] # dimension of word embedding
        
    def fit(self,X,y):
        return self
    
    def transform(self,X):
        return np.array([np.mean(text,axis=0) for text in X])
    
    
class TfidfEmbeddingVectorizer(object):
    def __init__(self, w2v):
        self.word2vec = w2v 
        self.word2weight = None
        self.dim = list(w2v.items())[0][1].shape[0] 

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x) # since the texts are already tokenized, we don’t process it in any way
        tfidf.fit(X)
        max_idf = max(tfidf.idf_) # if the word did not occur in X, it will be no less rare,
                                   # than the rarest word in the dict, therefore tf_idf of such a word is maximum
        self.word2weight = defaultdict(lambda: max_idf,[(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        
        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w] # the average weighted vector of words in the document
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0) # if all words are not from the w2v dict, then the average vector is zero
                for words in X
            ])