# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 19:45:37 2022

@author: ejo17
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf 
from tensorflow import keras

from nltk.corpus import stopwords
#from wordcloud import WordCloud
import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.pipeline import TransformerMixin
from sklearn.base import BaseEstimator

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import make_pipeline


import os

#Creating a dataframe with pandas and reading in the csv - also choosing specific columns with (2,5)
df = pd.read_csv('C:\\Users\\ejo17\\Desktop\\Uni work\\Year 3\\Project\\dataset.csv', header=None)
df = df[[2,5]]
df.columns = ['Tweet', 'Sentiment']
df.head()

#Creating a plot of the sentiment column as a visual output
sns.countplot(df['Sentiment'])

def clean_text(s):
    s = re.sub(r'http\S+', '', s)
    s = re.sub('(RT|via)((?:\\b\\W*@\\w+)+)', ' ', s)
    s = re.sub(r'@\S+', '', s)
    s = re.sub('&amp', ' ', s)
    return s
df['clean_tweet'] = df['Tweet'].apply(clean_text)

text = df['clean_tweet'].to_string().lower()    

# Encode Categorical Variable
X = df['clean_tweet']
# y = pd.get_dummies(df['label']).values
encode_cat = {"label":     {"['neutral']": 0, "['positive']": 1, "['negative']": 2},
             }
y_df = df.replace(encode_cat)
y = y_df['Sentiment']
y.value_counts()

seed = 101 # fix random seed for reproducibility
np.random.seed(seed)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=seed)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

vocab_size = 20000  # Max number of different word, i.e. model input dimension
maxlen = 80  # Max number of words kept at the end of each text

class TextsToSequences(Tokenizer, BaseEstimator, TransformerMixin):
    """ Sklearn transformer to convert texts to indices list 
    (e.g. [["the cute cat"], ["the dog"]] -> [[1, 2, 3], [1, 4]])"""
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        
    def fit(self, texts, y=None):
        self.fit_on_texts(texts)
        return self
    
    def transform(self, texts, y=None):
        return np.array(self.texts_to_sequences(texts))
        
sequencer = TextsToSequences(num_words=vocab_size)

class Padder(BaseEstimator, TransformerMixin):
    """ Pad and crop uneven lists to the same length. 
    Only the end of lists longernthan the maxlen attribute are
    kept, and lists shorter than maxlen are left-padded with zeros
    
    Attributes
    ----------
    maxlen: int
        sizes of sequences after padding
    max_index: int
        maximum index known by the Padder, if a higher index is met during 
        transform it is transformed to a 0
    """
    def __init__(self, maxlen=500):
        self.maxlen = maxlen
        self.max_index = None
        
    def fit(self, X, y=None):
        self.max_index = pad_sequences(X, maxlen=self.maxlen).max()
        return self
    
    def transform(self, X, y=None):
        X = pad_sequences(X, maxlen=self.maxlen)
        X[X > self.max_index] = 0
        return X

padder = Padder(maxlen)


batch_size = 128
max_features = vocab_size + 1

tf.random.set_seed(seed)

def create_model(max_features):
    """ Model creation function: returns a compiled LSTM"""
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

sklearn_lstm = KerasClassifier(build_fn=create_model, epochs=2, batch_size=batch_size, 
                               max_features=max_features, verbose=1)

# Build the Scikit-learn pipeline
pipeline = make_pipeline(sequencer, padder, sklearn_lstm)

pipeline.fit(X_train, y_train);

print('Computing predictions on test set...')

y_preds = pipeline.predict(X_test)