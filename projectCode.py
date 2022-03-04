# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 19:45:37 2022
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf 
from tensorflow import keras
import lime

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

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


from lime.lime_text import LimeTextExplainer


import os


#df = pd.read_csv('C:\\Users\\ejo17\\Desktop\\Uni work\\Year 3\\Project\\bitcointweets.csv', header=None)
#df = df[[1,7]]
#df.columns = ['tweet', 'label']
#df.head()

#Creating a dataframe with pandas and reading in the csv - also choosing specific columns with ['Tweet', 'Sentiment']
df = pd.read_csv('C:\\Users\\ejo17\\Desktop\\Uni work\\Year 3\\Project\\dataset.csv', header=0)
df = df[['Tweet','Sentiment']]

#head() returns the first n rows - default 5
df.head()

#Creating a plot of the sentiment column as a visual output
#sns.countplot(df['label'])
sns.countplot(df['Sentiment'])

#df['text_length'] = df['tweet'].apply(len)
#df[['label','text_length','tweet']].head()

df['text_length'] = df['Tweet'].apply(len)
df[['Sentiment','text_length','Tweet']].head()

def clean_text(s):
    s = re.sub(r'http\S+', '', s)
    s = re.sub('(RT|via)((?:\\b\\W*@\\w+)+)', ' ', s)
    s = re.sub(r'@\S+', '', s)
    s = re.sub('&amp', ' ', s)
    return s

#df['clean_tweet'] = df['tweet'].apply(clean_text)
df['clean_tweet'] = df['Tweet'].apply(clean_text)

text = df['clean_tweet'].to_string().lower()    

# Encode Categorical Variable
X = df['clean_tweet']
# y = pd.get_dummies(df['label']).values
#encode_cat = {"label":     {"['neutral']": 0, "['positive']": 1, "['negative']": 2},
#             }

#converting categorical variable into dummy values
y = pd.get_dummies(df['Sentiment']).values

#categorising the strings to ints and replacing them with the chosen values
encode_cat = {"Sentiment":     {"NEITHER": 0, "POSITIVE": 1, "NEGATIVE": 2}}
y_df = df.replace(encode_cat)


#y = y_df['label']
y = y_df['Sentiment']
y.value_counts()

seed = 101 # fix random seed for reproducibility
np.random.seed(seed)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

vocab_size = 20000  # Max number of different word, i.e. model input dimension
maxlen = 80  # Max number of words kept at the end of each text


class TextsToSequences(Tokenizer, BaseEstimator, TransformerMixin):
    """Sklearn transformer to convert texts to indices list 
    (e.g. [["the cute cat"], ["the dog"]] -> [[1, 2, 3], [1, 4]]) """
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        
    def fit(self, texts, y=None):
        self.fit_on_texts(texts)
        return self
    
    def transform(self, texts, y=None):
        return np.array(self.texts_to_sequences(texts), dtype=object)
        
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
    """ Model creation function: returns a compiled LSTM """
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



def model_evaluate(): 
    
    print('Test Accuracy:\t{:0.1f}%'.format(accuracy_score(y_test,y_preds)*100))
    
    #classification report
    print('\n')
    print(classification_report(y_test, y_preds))

    #confusion matrix
    confmat = confusion_matrix(y_test, y_preds)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    
    
model_evaluate()

idx = 26
test_text = np.array(X_test)
test_class = np.array(y_test)
text_sample = test_text[idx]



#class_names = ['neutral', 'positive', 'negative']
class_names = ['NEITHER', 'POSITIVE', 'NEGATIVE']
print(text_sample)
print('Probability =', pipeline.predict_proba([text_sample]).round(3))
print('True class: %s' % class_names[test_class[idx]])


explainer = LimeTextExplainer(class_names=class_names)
exp = explainer.explain_instance(text_sample, pipeline.predict_proba, num_features=6, top_labels=2)
#outputting the explainer as a figure
exp.as_pyplot_figure()

#removing the word successful from the second sample to see the difference it makes 


#print('Probability =', pipeline.predict_proba([text_sample2]).round(3))