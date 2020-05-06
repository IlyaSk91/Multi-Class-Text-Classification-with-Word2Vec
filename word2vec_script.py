import pandas as pd 
from gensim.models import Word2Vec 
from preprocessing import split_file,clean_text,remove_stopwords,func_lemma,func_container,func_tokenize #preprocessing functions
from vectorizer import MeanEmbeddingVectorizer,TfidfEmbeddingVectorizer
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from gensim.models.word2vec import FAST_VERSION 
FAST_VERSION=1

import sys
sys.path.append('../lib/')

file=[]
path=r'rvm.txt' # path to file
for string in open(path,'r',encoding='cp1251'):
    file.append(string.lower())
    

file_split=split_file(file)
text=clean_text([file_split[i][0] for i in range(len(file_split))]) # remove symbols in text
clear_text=remove_stopwords(text) # remove stop-words in text
s=func_lemma(func_container(clear_text)) # lemmatization procedure
w=func_tokenize(s) # w train dataset after preprocessing procedure


path=r'lenta-ru-news.csv' # path to test dataset
df = pd.read_csv(path,engine='python', delimiter=',',encoding = "utf-8-sig")

# plot topic news distribution
y_pos=np.arange(len(df['topic'].value_counts()))
performance=df['topic'].value_counts()
plt.figure(figsize=(8,6))
plt.bar(y_pos,performance,align='center',alpha=0.5,color='g',width=0.8)
plt.xticks(y_pos,df['topic'].value_counts().index.tolist(),rotation=90,size=15)
plt.yticks(size=15)
plt.xlabel('Topics',size=15)
plt.ylabel('Number of news',size=15)
plt.show()

test_dataset=[]
for text in df['text'][:100]: # take only 100 news from corpus
    test_text=clean_text(text)
    test_text=[''.join(word for word in test_text if not word=='')]
    test_text=remove_stopwords(test_text)
    test_text=func_lemma(test_text)
    test_text=func_tokenize(test_text)
    test_dataset.append(*test_text)
    
# create and train word2vec model    
model = Word2Vec(w,min_count=2,workers=6,size=300,window=3,hs=0,sg=1)

# model_dict return word in vocab and vector
model_dict={}
for text in test_dataset:
    for word in text:  
        word_vectors = model.wv
        if word in word_vectors.vocab: # if word in vocab
            model_dict.update({word:model[word]}) # add to dict
        else:
            continue
            
transform_corpus=[]
for text in test_dataset:
    transform_corpus.append([model[word] for word in text if word in model.wv.vocab])

# list of classification models
clfs=[LogisticRegression(multi_class='multinomial',solver='sag', max_iter=1000),SVC(kernel='poly', gamma='scale', coef0=1, degree=3),RandomForestClassifier(n_estimators=400)]

def classification_models(X,y):
    ans=[]
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,shuffle=True)
    for number,method in enumerate(clfs):
        method.fit(X_train,y_train)
        ans.append('Classifier %d: accuracy score: %.3f' % (number,accuracy_score(y_test,method.predict(X_test))))
    return ans

X=MeanEmbeddingVectorizer(model_dict).transform(transform_corpus) # mean of all word vectors in each topic
classification_models(X,df['topic'][:100])

X1=TfidfEmbeddingVectorizer(model_dict).fit(test_dataset,df['topic'][:100]).transform(test_dataset) # use TF-IDF for all words in each topic
classification_models(X1,df['topic'][:100])

# dimensionality reduction
X = model[list(model_dict.keys())[:8]] # take only first 8 words
y=model[df.loc[0,'topic'].lower()] 
data=np.vstack((X,y)) # join words from text and topic
pca = PCA(n_components=2) # 2 components
result = pca.fit_transform(X)
colors=[*(['b']*7),'r'] 
# output results
plt.scatter(result[:, 0], result[:, 1],c=colors)
words = list(model_dict.keys())[:8]

for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]),size=12)

plt.xticks(size=15)
plt.yticks(size=15)
plt.xlabel('X1 component',size=15)
plt.ylabel('X2 component',size=15)
plt.show()

