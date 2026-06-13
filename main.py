# load the model and libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


#load imdb word index
word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

#load the pre-trained model
model=load_model('simple_rnn.h5')

def decode_review(encoded_review):#function to decode reviews
  return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])


def preprocess_text(text):#function to preprocess user input
  words=text.lower().split()
  encoded_review=[word_index.get(word,2)+3 for word in words]
  padded_review=pad_sequences([encoded_review],maxlen=500)
  return padded_review

def predict_sentiment(text):#function to predict sentiment

  preprocessed_input=preprocess_text(text)

  prediction=model.predict(preprocessed_input,verbose=0)

  sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'

  return sentiment,prediction[0][0]



#streamlit app
import streamlit as st 
st.title('IMDB movie review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

user_input=st.text_area('Enter your review here')

if st.button('Predict'):
  if user_input:
    sentiment,score=predict_sentiment(user_input)
    st.write(f'Sentiment:{sentiment}')
    st.write(f'Score:{score}')

    st.write(f'Sentiment:{sentiment}')
    st.write(f'Prediction Score:{score}')
  else:
    st.write('Please enter a review')

    


