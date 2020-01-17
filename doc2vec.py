from gensim.models.doc2vec import Doc2Vec
from numpy import linalg as LA
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np

d2v= Doc2Vec.load("doc2vec_100dim_science.model")

def Doc2vec(doc):
    test_data = word_tokenize(doc.lower())
    return d2v.infer_vector(test_data)

def cosin(v1,v2):
    if(LA.norm(v1)!=0 and LA.norm(v2)!=0):
        return (np.dot(np.array(v1),np.array(v2))/(LA.norm(v1) * LA.norm(v2)))
    else:
        return 1

df = pd.read_csv("Losstesttmp.csv")
df = df['val']

doc1 = Doc2vec(df[0])
doc2 = Doc2vec(df[1])
print(cosin(doc1, doc2))
