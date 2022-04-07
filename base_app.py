# Streamlit dependencies
import streamlit as st
import joblib,os

## data dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
#from nlppreprocess import NLP # pip install nlppreprocess
#import en_core_web_sm
from nltk import pos_tag

import seaborn as sns
import re

from nlppreprocess import NLP
nlp = NLP()

def cleaner(line):
    
    # Removes RT, url and trailing white spaces
    line = re.sub(r'^RT ','', re.sub(r'https://t.co/\w+', '', line).strip()) 

    # Removes puctuation
    punctuation = re.compile("[.;:!\'’‘“”?,\"()\[\]]")
    tweet = punctuation.sub("", line.lower()) 

    # Removes stopwords
    nlp_for_stopwords = NLP(replace_words=True, remove_stopwords=True, remove_numbers=True, remove_punctuations=False) 
    tweet = nlp_for_stopwords.process(tweet) # This will remove stops words that are not necessary. The idea is to keep words like [is, not, was]
    # https://towardsdatascience.com/why-you-should-avoid-removing-stopwords-aa7a353d2a52

    # tokenisation
    # We used the split method instead of the word_tokenise library because our tweet is already clean at this point
    # and the twitter data is not complicated
    tweet = tweet.split() 

    # POS 
    pos = pos_tag(tweet)


    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tweet = ' '.join([lemmatizer.lemmatize(word, po[0].lower()) if po[0].lower() in ['n', 'r', 'v', 'a'] else word for word, po in pos])
    # tweet = ' '.join([lemmatizer.lemmatize(word, 'v') for word in tweet])

    return tweet



# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

#https://github.com/QAInsights/Streamlit-JMeter/blob/master/about.py
#https://github.com/QAInsights/Streamlit-JMeter

#######wordcloud
#https://github.com/rezaho/streamlit-wordcloud

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	st.sidebar.title("Navigation")

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifier")
 
	from PIL import Image
	image = Image.open('resources/imgs/twitter_mask.png')
	st.image(image, caption='Which Tweet are you?', use_column_width=True)
 
	# st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	pages = ["Information", "Exploratory Data Analysis", "Classify Tweet", "App Developers"]
	# selection = st.sidebar.selectbox("Choose Option", pages)
 	

	st.subheader('Climate Change Belief Analysis: Based on Tweets')
	 ##creating a sidebar for selection purposes
	selection = st.sidebar.radio("Go to....", pages)
 

	




	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("""We have deployed Machine Learning models that are able to classify whether or not a person believes in climate change, based on their novel tweet data. Our models provide robust solutions that can give companies access to a broad base of consumer sentiment, spanning multiple demographic and geographic categories - thus increasing their insights and informing future marketing strategies. 
Explore the app
""")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message', 'tweetid']]) # will write the df to the page
	if selection == "Exploratory Data Analysis":
		st.info("The following are some of the steps and visual outcomes of the exploring the raw data.")
		st.write("Distribution of messages per sentiment")
		# Labeling the target
		raw['sentiment'] =[['Negative', 'Neutral', 'Positive', 'Factual'][x+1] for x in raw['sentiment']]
          # checking the distribution
		st.write('The numerical proportion of the sentiments')
		values = raw['sentiment'].value_counts()/raw.shape[0]
		labels = (raw['sentiment'].value_counts()/raw.shape[0]).index
		colors = ['lightgreen', 'blue', 'purple', 'lightsteelblue']
		fig, ax = plt.subplots()
  		
		ax.pie(x=values, labels=labels, autopct='%1.1f%%', startangle=90, explode= (0.04, 0, 0, 0), colors=colors)
		st.pyplot(fig)
  
        #   # checking the distribution
		fig, ax = plt.subplots()
		ax.set_title('Distribution of messages per sentiment')
		ax.set_xlabel('Sentiment')
		ax.set_ylabel('Count')
		ax = sns.countplot(x='sentiment', data=raw)
		st.pyplot(fig)
  
		#create and generate a wordcloud
		fig, ax = plt.subplots()
		wordcloud = WordCloud(background_color='white', max_words=100, max_font_size=60).generate(' '.join(raw['message']))
		ax.set_title('Wordcloud of messages')
		ax = plt.imshow(wordcloud, interpolation="bilinear")
		plt.show()
		st.pyplot(fig)


					
	if selection == "Classify Tweet":
		st.info('Make Predictions of your Tweet(s) using our ML Model')
		data_source = ['Select option', 'Single text', 'Dataset'] ## differentiating between a single text and a dataset inpit

		source_selection = st.selectbox('What to classify?', data_source)

				# Load Our Models
		def load_prediction_models(model_file):
			loaded_models = joblib.load(open(os.path.join(model_file),"rb"))
			return loaded_models

				# Getting the predictions
		def get_keys(val,my_dict):
			for key,value in my_dict.items():
				if val == value:
					return key


			if source_selection == 'Single text':
				### SINGLE TWEET CLASSIFICATION ###
				st.subheader('Single tweet classification')

				input_text = st.text_area('Enter Text (max. 140 characters):') ##user entering a single text to classify and predict
				all_ml_models = ["LR","NB","RFOREST","DECISION_TREE"]
				model_choice = st.selectbox("Choose ML Model",all_ml_models)



				prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}
				if st.button('Classify'):

					st.text("Original test ::\n{}".format(input_text))
					text1 = cleaner(input_text) ###passing the text through the 'cleaner' function
					vect_text = tweet_cv.transform([text1]).toarray()
					if model_choice == 'LR':
						predictor = load_prediction_models("resources/Logistic_regression.pkl")
						prediction = predictor.predict(vect_text)
						# st.write(prediction)
					elif model_choice == 'RFOREST':
						predictor = load_prediction_models("resources/RFOREST_model.pkl")
						prediction = predictor.predict(vect_text)
						# st.write(prediction)
					elif model_choice == 'NB':
						predictor = load_prediction_models("resources/NB_model.pkl")
						prediction = predictor.predict(vect_text)
						# st.write(prediction)
					elif model_choice == 'DECISION_TREE':
						predictor = load_prediction_models("resources/DTrees_model.pkl")
						prediction = predictor.predict(vect_text)
					# st.write(prediction)

					final_result = get_keys(prediction,prediction_labels)
					st.success("Tweet Categorized as:: {}".format(final_result))
			if source_selection == 'Dataset':
				### DATASET CLASSIFICATION ###
				st.subheader('Dataset tweet classification')

				all_ml_models = ["LR","NB","RFOREST","SupportVectorMachine", "MLR", "LDA"]
				model_choice = st.selectbox("Choose ML Model",all_ml_models)

				st.info('for more information on the above ML Models please visit: https://datakeen.co/en/8-machine-learning-algorithms-explained-in-human-language/')


				prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}
				text_input = st.file_uploader("Choose a CSV file", type="csv")
				if text_input is not None:
					text_input = pd.read_csv(text_input)

				#X = text_input.drop(columns='tweetid', axis = 1, inplace = True)   

				uploaded_dataset = st.checkbox('See uploaded dataset')
				if uploaded_dataset:
					st.dataframe(text_input.head(25))

				col = st.text_area('Enter column to classify')

				#col_list = list(text_input[col])

				#low_col[item.lower() for item in tweet]
				#X = text_input[col]

				#col_class = text_input[col]
				
				if st.button('Classify'):

					st.text("Original test ::\n{}".format(text_input))
					X1 = text_input[col].apply(cleaner) ###passing the text through the 'cleaner' function
					vect_text = tweet_cv.transform([X1]).toarray()
					if model_choice == 'LR':
						predictor = load_prediction_models("resources/Logistic_regression.pkl")
						prediction = predictor.predict(vect_text)
						# st.write(prediction)
					elif model_choice == 'RFOREST':
						predictor = load_prediction_models("resources/Random_model.pkl")
						prediction = predictor.predict(vect_text)
						# st.write(prediction)
					elif model_choice == 'NB':
						predictor = load_prediction_models("resources/NB_model.pkl")
						prediction = predictor.predict(vect_text)
						# st.write(prediction)
					elif model_choice == 'SupportVectorMachine':
						predictor = load_prediction_models("resources/svm_model.pkl")
						prediction = predictor.predict(vect_text)

					elif model_choice == 'MLR':
						predictor = load_prediction_models("resources/mlr_model.pkl")
						prediction = predictor.predict(vect_text)

					elif model_choice == 'SupportVectorMachine':
						predictor = load_prediction_models("resources/simple_lda_model.pkl")
						prediction = predictor.predict(vect_text)

                
					# st.write(prediction)
					text_input['sentiment'] = prediction
					final_result = get_keys(prediction,prediction_labels)
					st.success("Tweets Categorized as:: {}".format(final_result))

					
					csv = text_input.to_csv(index=False)
					b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
					href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'

					st.markdown(href, unsafe_allow_html=True)
		
     
	# Building out the predication page
	if selection == "Classify":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

        

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
