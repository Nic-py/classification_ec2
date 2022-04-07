# Importing modules for data science and visualization
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# NLP Libraries
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nlppreprocess import NLP
from nltk import pos_tag
import pickle
import json

# reading in the dataset
train = pd.read_csv("resources/train.csv")

# Data cleaning for furthur sentiment analysis

def cleaner(line):
    '''
    For preprocessing the data, we regularize, transform each upper case into lower case, tokenize,
    normalize and remove stopwords. Normalization transforms a token to its root word i.e. 
    These words would be transformed from "love loving loved" to "love love love."
    
    '''
    
    # print("Original:\n", line, '\n'*2)

    # Removes RT, url and trailing white spaces
    line = re.sub(r'^RT ','', re.sub(r'https://t.co/\w+', '', line).strip()) 
    # print("After removing RT, url and trailing white spaces:\n" + line, '\n'*2)

    emojis = re.compile("["
                           u"\U0001F600-\U0001F64F"  # removes emoticons,
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

    line = emojis.sub(r'', line)
    # Removes puctuation
    punctuation = re.compile("[.;:!\'’‘“”?,\"()\[\]]")
    tweet = punctuation.sub("", line) 
    # print("After puctuation:\n", tweet, '\n'*2)

    # Removes stopwords
    nlp_for_stopwords = NLP(replace_words=True, remove_stopwords=True, 
                            remove_numbers=True, remove_punctuations=False) 
    tweet = nlp_for_stopwords.process(tweet) # This will remove stops words that are not necessary. The idea is to keep words like [is, not, was]
    # print("After removing stopwords:\n", tweet, '\n'*2)

    # tokenisation
    # We used the split method instead of the word_tokenise library because our tweet is already clean at this point
    # and the twitter data is not complicated
    tweet = tweet.split() 
    # print("After tokenisation:\n", tweet, '\n'*2)

    # POS 
    pos = pos_tag(tweet)
    # print("After POS:\n", pos, '\n'*2)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tweet = ' '.join([lemmatizer.lemmatize(word, po[0]) 
                      if (po[0] in ['n', 'r', 'v', 'a'] and word[0] != '@') else word for word, po in pos])
    # print("After Lemmatization:\n", tweet, '\n'*2)

    return tweet

# Cleaning the dataset using the above function
cleaned_text = train['message'].apply(cleaner)

# Pickle tool for use within our app
save_path = 'resources/preprocessor.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(cleaned_text, open(save_path,'wb'))