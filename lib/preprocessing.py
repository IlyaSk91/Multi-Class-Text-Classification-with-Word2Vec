import re 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from pymystem3 import Mystem 


# Text split function
def split_file(file):
    file_split=[]
    file_split=[re.split('\t',string,maxsplit=1) for string in file] 
    return file_split
                 
# pattern for regex
pattern=re.compile(r'[—...,!?"-;\n\t\d><]')

#Clean function
def clean_text(text):
    text=[re.sub(pattern,'',string) for string in text] # remove symbols in pattern and replace ''
    return text

#remove stop-words
def remove_stopwords(text):
    stop_words = set(stopwords.words('russian')) #use russian dictionary
    removed_stopwords=[]
    for i in range(len(text)):
        # concatenate all words if the are not in stop-words
        removed_stopwords.append(' '.join([word for word in text[i].split() if not word in stop_words]))
    return removed_stopwords

#Concatenate text
def func_container(text):
    clean_text=[]
    i=0
    for sen in text:
        if sen.startswith('rvm') and i==0: # if word start with 'rvm' and i=0
            s='' 
            i=1
        elif sen.startswith('rvm') and i==1: # if word start with 'rvm' and i=1
            clean_text.append(s)
            i=1
            s=''
        else:
            s=' '.join((s,sen)) # join current word with others
    return clean_text

#Lemmatization function
def func_lemma(text):
    m=Mystem() # lemmatization model
    lemmas=[]
    for i in range(len(text)):
        # join all words in i-topic after lemmatization
        lemmas.append(''.join(m.lemmatize(text[i]))) 
    return lemmas

#Split text into tokens
def func_tokenize(text):
   tokens=[]
   for i in range(len(text)):
       tokens.append(word_tokenize(text[i])) 
   return tokens
