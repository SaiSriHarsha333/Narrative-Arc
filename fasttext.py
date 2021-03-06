# -*- coding: utf-8 -*-
"""fasttext

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_UXl85Rfrk7mfNB0J6m33L0KWtCjZPuR
"""

from google.colab import drive
drive.mount('/content/gdrive')
# drive.mount("/content/gdrive", force_remount=True)

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/gdrive/My Drive/WSL/

!pip install -U gensim

import pandas as pd
import numpy as np
from gensim.models import FastText
import spacy, string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pickle
import nltk

df = pd.read_csv("allcontent_required_NotNull_sum_outCaptions.csv")
# df = df.drop_duplicates(subset=['resource_id'], keep='first')
df = df[pd.notnull(df['val'])]
# df['description'] = df['title'] + " " + df['description']
df=df['val']

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
nltk.download('wordnet')

nlp = spacy.load('en_core_web_sm')
stopwords = set(stopwords.words('english'))
punctuation = set(string.punctuation)
lemmatize = WordNetLemmatizer()
words = set(nltk.corpus.words.words())

sentences = []
for item in df:
        # gensim.utils.simple_preprocess(item, deacc=True)
        tokens = []
        tokens.append('<start>')
        tokens.extend(nltk.tokenize.word_tokenize(str(item).lower()))
        tokens.append('<end>')
        sentences.append(tokens)

sentences.append(['<unk>'])

model_ted = FastText(sentences, size=256, window=5, min_count=0, sg=1, iter=150, compatible_hash=True)

file = open('fasttext.model', 'wb')
pickle.dump(model_ted, file)
file.close()

model_ted.wv.vocab['<unk>'].index