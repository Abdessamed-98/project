from flask import Flask, redirect, request, render_template
#import time
'''
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import pickle

import joblib
from scipy.sparse import hstack

from keras import models
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
'''
### WEB APP
app = Flask(__name__)
questions = []
first = True
@app.route('/', methods=['POST', 'GET'])
def prediction():
    if request.method == 'POST':
        global questions, first
        question = request.form['question']
        m = request.form['model']
        questions = pred(question, m)
        first = False
        return redirect('/')
    else:
        return render_template("interface.html" , data = {'questions':questions, 'first':first})

def pred(question, model):
    #time.sleep(10)
    if question == 'hello':
        if model == 'siamese':
            return ['slm','hi','hola']
        else:
            return ['ss','hey','bj']
    else:
        return []

"""
###############     nlp section     ###############
### functions needed
#custom function for model
def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))
    
#load vocabulary function
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

#preprocessing function
def clean(text):
    text = str(text)
    text = text.lower()
    
    text = ' '.join([word for word in text.split() if not word in stops])
    
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    return text

def text_to_word_list(text):
    '''convert texts to a list of words '''
    text = clean(text)
    text = text.split()
    return text

#text_to_id function
def text_to_id(question):
    q2n = []  # q2n -> question numbers representation
    for word in text_to_word_list(question):
        if word in vocabulary:
            q2n.append(vocabulary[word])
    return q2n

#pad sequences
def pad_seq(x):
    q = [text_to_id(x)]
    q = pad_sequences(q, maxlen=105)
    return q

#custom_round
def custom_round(n, x):
    if n < x:
        return 0
    else:
        return 1

custom_round = np.vectorize(custom_round)

### load section
#load Siamese model
myModel = models.load_model('myModel', custom_objects = { "exponent_neg_manhattan_distance": exponent_neg_manhattan_distance })

#load vocabulary
vocabulary = load_obj("vocabulary" )

# Load KNN model
knn = joblib.load('knn.pkl')

#load TFIDF
tfidf = pickle.load(open('tfidf.pickle', 'rb'))

#load stopwords
stops = set(stopwords.words('english'))

#load dataset
df = pd.read_csv("workDataset.csv")
df = df.head(100)

### prediction function
def pred(text, choice):
    questions = []

    if choice == 'siamese':
        ### w2v Siamese ###
        text =  pad_seq(text)
        a = df['question1'].apply(lambda x: myModel.predict([pad_seq(x), text], verbose=0)[0][0])
        a = a[custom_round(a, 0.6) == 1]
        index = a.nlargest(3).index.values.tolist()
        for i in index:
            questions.append(df["question1"][i])
    else:
        ### tfidf knn ###
        t = []
        text = tfidf.transform([clean(text)])
        b = df['question1'].apply(lambda x: knn.predict_proba(hstack((tfidf.transform([clean(x)]),text)))[0])
        for x in b:
            t.append(x[1])
        t = pd.Series(t)
        b = t[t.round() == 1]
        index = b.nlargest(3).index.values.tolist()
        for i in index:
            if len(questions) < 3:
                questions.append(df["question1"][i])
            else:
                break
    return questions

###################################################
"""

if __name__ == '__main__':
    app.run(debug = True)