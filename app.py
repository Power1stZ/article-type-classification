import joblib
#for text pre-processing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import numpy as np
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

import streamlit as st
from wordcloud import WordCloud
from collections import Counter

#for word embedding
import gensim
from gensim.models import Word2Vec


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))
    def fit(self, X, y):
            return self
    def transform(self, X):
            return np.array([
                np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                        or [np.zeros(self.dim)], axis=0)
                for words in X
            ])

#convert to lowercase, strip and remove punctuations
@st.cache(suppress_st_warning=True)
def preprocess(text):
    text = text.lower() 
    text=text.strip()  
    text=re.compile('<.*?>').sub('', text) 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text) 
    return text

 
# STOPWORD REMOVAL
@st.cache(suppress_st_warning=True)
def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)

# LEMMATIZATION
# Initialize the lemmatizer
wl = WordNetLemmatizer()
 
# This is a helper function to map NTLK position tags
@st.cache(suppress_st_warning=True)
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Tokenize the sentence
@st.cache(suppress_st_warning=True)
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)

@st.cache(suppress_st_warning=True)
def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))

category = [
            'Science', 
            'Food', 
            'VideoGames', 
            'News', 
            'Informative', 
            'Automobile', 
            'Tech', 
            'Blog', 
            'Entertainment,Comedy', 
            'Comedy,Entertainment', 
            'Automobile,Comedy', 
            'Blog,Comedy', 
            'Comedy,Informative', 
            'Entertainment', 
            'Entertainment,Blog', 
            'Comedy', 
            'Tech,News', 
            'Tech,Informative', 
            'Blog,Entertainment', 
            'Tech,Comedy', 
            'Food,Entertainment', 
            'Blog,Science'
          ]

@st.cache(suppress_st_warning=True)
def generate_wordcloud(word):
    wc = WordCloud(
        width=1320,
        height=720,
        background_color="white",
        scale=1
    )
    wc.fit_words(word)
    return wc

st.header('Article Classification')

with st.form("Select Model"):
    st.session_state.model = st.selectbox('Select Model',['MaxEnt + TFIDF','MaxEnt + Word2Vec','Naive Bayes + TFIDF'])
    st.session_state.text = st.text_area('Input Article')
    st.session_state.submit = st.form_submit_button('Classify')
with st.spinner('Processing'):
    if st.session_state.model == 'MaxEnt + TFIDF':
        lr = joblib.load('./models/lr_tfidf.pkl')
        tfidf = joblib.load('./models/tfidfvectorizer.pkl')
        rawinput = finalpreprocess(st.session_state.text)
        input = tfidf.transform([rawinput])
        st.session_state.category = category[lr.predict(input)[0]]
    elif st.session_state.model == 'MaxEnt + Word2Vec':
        lr_w2v = joblib.load('./models/lr_w2v.pkl')
        model = joblib.load('./models/word2vec.pkl')
        w2v = dict(zip(model.wv.index2word, model.wv.syn0)) 
        modelw2v = MeanEmbeddingVectorizer(w2v)
        rawinput = finalpreprocess(st.session_state.text)
        input = nltk.word_tokenize(rawinput)
        input = modelw2v.transform([input])
        st.session_state.category = category[lr_w2v.predict(input)[0]]
    else:
        nb = joblib.load('./models/nb_tfidf.pkl')
        tfidf = joblib.load('./models/tfidfvectorizer.pkl')
        rawinput = finalpreprocess(st.session_state.text)
        input = tfidf.transform([rawinput])
        st.session_state.category = category[nb.predict(input)[0]]

if str(st.session_state.text).split() != []:
    word = rawinput.split()
    wc = generate_wordcloud(Counter(word))
    st.image(wc.to_array(),use_column_width=True)
    st.subheader('This Article Type is " ' + st.session_state.category + ' "')
else:
    st.subheader('Please fill the Article !!')