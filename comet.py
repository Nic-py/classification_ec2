# import comet_ml in the top of your file
from comet_ml import Experiment

# Adding the following code to our machine learning file
experiment = Experiment(api_key="9gsTl4Wv73PDkYEoX8PUt5RSX",
                        project_name="NLP", workspace="ms-noxolo")

# Run your code and go to https://www.comet.ml/

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
# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.metrics import accuracy_score

# Loading in the datasets
train = pd.read_csv("train.csv").fillna(' ')
test = pd.read_csv("test.csv").fillna(' ')
sample_submission = pd.read_csv('sample_submission.csv')

def data_preprocessor(df):
    stop_words = set(stopwords.words('english'))
    lemm = WordNetLemmatizer()
    Tokenized_Doc=[]
    print("Preprocessing data.........\n")
    for data in df['message']:
        review = re.sub('[^a-zA-Z]', ' ', data)
        url = re.compile(r'https?://\S+|www\.\S+')
        review = url.sub(r'',review)
        html=re.compile(r'<.*?>')
        review = html.sub(r'',review)
        emojis = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
        review = emojis.sub(r'',review)
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(review)
        gen_tweets = [lemm.lemmatize(token) for token in tokens if not token in stop_words]
        Tokenized_Doc.append(gen_tweets)
        df['tweet tokens'] = pd.Series(Tokenized_Doc)
        
    return df

# Cleaning the data
train_df = data_preprocessor(train)
test_df = data_preprocessor(test)

# defining features and the label
X = train_df['tweet tokens']
y = train_df['sentiment']

# Tranforming the dataset
data = train_df['tweet tokens']
corpus = [' '.join(i) for i in data] #create your corpus here
vectorizer=TfidfVectorizer(use_idf=True, max_df=0.95)
X = vectorizer.fit_transform(corpus)

# Spliting the datasets and training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scikit_log_reg = LogisticRegression(solver='liblinear',random_state=42) #, C=5, penalty='l2',max_iter=1000)
model=scikit_log_reg.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluating the model performance
from sklearn import metrics
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

# Predicting on the test.csv
data2 = test_df['tweet tokens']
corpus = [' '.join(i) for i in data2]
tests = vectorizer.transform(corpus, copy=True)
#print(vectorizer.get_feature_names())
#print(tests.toarray())

pred = model.predict(tests)
predictions = pred[:]

