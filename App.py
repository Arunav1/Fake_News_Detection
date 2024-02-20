import streamlit as st
import numpy as np #provides arrays
import pandas as pd #for data cleaning
import re  #for making patterns
from nltk.corpus import stopwords # the in for of in with these are stop words in English literature.
from nltk.stem.porter import PorterStemmer  # finding the stem word e.g. Loving Loved == Love
from sklearn.feature_extraction.text import TfidfVectorizer # converting the word into vector e.g. Love == [0,0]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

news_df = pd.read_csv('fake_news_train_clean.csv')

#stemming function:
ps = PorterStemmer()

def stemming(title_txt):
    stemmed_content = re.sub('[^a-zA-Z]',' ', title_txt)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content

#vectorization:
X = news_df['title_txt'].values
Y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

#split data into train and test data:
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify = Y, random_state=1)

#fitting logistic regression model into the training data:
model = LogisticRegression()
model.fit(X_train, y_train)

#web app using streamlit:
st.title("Fake News Detection")
input_text = st.text_input("Enter News Article")

def prediction(input_text):
    input_data = vector.transform([input_text])
    predicts = model.predict(input_data)
    return predicts[0]

if input_text:
    pred = prediction(input_text)
    
    if pred == 1:
        st.write("The news is Real")
    else:
        st.write("The news is Fake")    