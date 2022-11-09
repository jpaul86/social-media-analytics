import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim import downloader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import helper

################################################################
################ Rule-based sentiment analsis ##################
################################################################

sentences = ["VADER is smart, handsome, and funny.",
             "VADER is smart, handsome, and funny!",
             "VADER is very smart, handsome, and funny.",
             "VADER is VERY SMART, handsome, and FUNNY.",
             "VADER is VERY SMART, handsome, and FUNNY!!!",
             "VADER is VERY SMART, uber handsome, and FRIGGIN FUNNY!!!",
             "VADER is not smart, handsome, nor funny.",
             "The book was good.",
             "At least it isn't a horrible book.",
             "The book was only kind of good.",
             "The plot was good, but the characters are uncompelling and the dialog is not great.",
             "Today SUX!",
             "Today only kinda sux! But I'll get by, lol",
             "Make sure you :) or :D today!",
             "Catch utf-8 emoji such as such as 💘 and 💋 and 😁",
             "Not bad at all",
             ]

df = pd.DataFrame({'sentences': sentences})

vader = SentimentIntensityAnalyzer()
df['vader'] = df.sentences.apply(
    lambda x: vader.polarity_scores(x)['compound'])
df['textblob'] = df.sentences.apply(lambda x: TextBlob(x).polarity)
df


################################################################
################ Twitter data###################################
################################################################

# Read data
df = pd.read_csv("data/twitter_labeled.csv")
df = df.replace({'sentiment': {'pos': 1, 'neg': 0}})

# Vader
vader = SentimentIntensityAnalyzer()
df['vader_scores'] = df.text.apply(lambda x: vader.polarity_scores(x))
df['vader_compound'] = df.vader_scores.apply(lambda x: x['compound'])
df['vader_polarity'] = df.vader_compound.apply(lambda x: 1 if x >= 0 else 0)
pd.crosstab(df.sentiment, df.vader_polarity, normalize='all')
helper.evaluate(df.sentiment, df.vader_polarity)


TextBlob()
# TextBlob
df['textblob_score'] = df.text.apply(
    lambda tweet: TextBlob(tweet).sentiment[0])
df['textblob_pred'] = df.textblob_score.apply(lambda x: 1 if x >= 0 else 0)
pd.crosstab(df.sentiment, df.textblob_pred, normalize='all')
helper.evaluate(df.sentiment, df.textblob_pred)


################################################################
####### Machine-learning based sentiment analsis ###############
################################################################

###### A. Bag of Words Feature approach #######

# 1. Preprocess
df["normalized"] = df.text.apply(lambda x: helper.normalize_document(x))

# 2. Split data
corpus_train, corpus_test, y_train, y_test = train_test_split(
    df.normalized, df.sentiment, test_size=0.3, random_state=0)

# 3. Bag of Words Features
bow = CountVectorizer(min_df=2)
X_train = bow.fit_transform(corpus_train)
X_test = bow.transform(corpus_test)


# 4. Train logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)  # training the model

# 5. Predict on test data
y_logistic = model.predict(X_test)
pd.DataFrame(zip(y_test, y_logistic), columns=["y_test", "y_logistic"]).head()

# 6. Evaluate
pd.crosstab(y_test, y_logistic)
helper.evaluate(y_test, y_logistic)


###### B. Static Embeddings #######

# Load dictionary of word embeddings (trained on twitter data)
wv = downloader.load('glove-twitter-25')

# Create document embeddings

document_embeddings = helper.avg_embeddings_corpus(df.normalized, wv)

# Supervised Machine learning (as before)
X_train, X_test, y_train, y_test = train_test_split(
    document_embeddings, df.sentiment, test_size=0.3, random_state=0)
model = LogisticRegression()
model.fit(X_train, y_train)
y_wv_logistic = model.predict(X_test)
pd.crosstab(y_test, y_wv_logistic)
helper.evaluate(y_test, y_wv_logistic)
