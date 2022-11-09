# Packages
from locale import normalize
import gensim.downloader
import spacy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import re

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA


# Read corpus from file
with open('documents.txt') as f:
    corpus = f.read().splitlines()


################
# Normalize ####
################
stop_words = nltk.corpus.stopwords.words('english')
wpt = nltk.WordPunctTokenizer()

# We want to normalize it, get rid of some additional words
def normalize_document(doc):
    doc = re.sub(r'[^\w\s]', '', doc)
    doc = doc.lower()           # we lowercase everything
    doc = doc.strip()           # we strip white spaces
    tokens = wpt.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens)
    return doc


normalize_corpus = np.vectorize(normalize_document)
normalized_corpus = normalize_corpus(corpus)
normalized_corpus
# Now we'll have a list of clean documents. This is where we want to start.

################
# Bag of Words #
################
vectorizer = CountVectorizer()                       # Bad of Words
# vectorizer = CountVectorizer(ngram_range(1,2))    # here we could also take the bigrams with the 2. So we get unigrams and bigrams!
# vectorizer = CountVectorizer(ngram_range(1,2), min_df=3)    # mind_df=3 word should occur in a minimum of 3 docs, we eliminate typos this way!

vectorizer = TfidfVectorizer()                      # by default he normalizes stuff!
features = vectorizer.fit_transform(normalized_corpus)
feature_names = vectorizer.get_feature_names_out(features)
pd.DataFrame(features.toarray(), columns=feature_names, index=corpus)


################
# Similarities #
################
similarity = cosine_similarity(features)
sns.clustermap(similarity, yticklabels=corpus, xticklabels=corpus)

# Get the closest sentences
# Recommender: Given document what is the most similar other document in corupus
np.fill_diagonal(similarity, np.nan)
closest = np.nanargmax(similarity, axis=0)
pd.DataFrame([(corpus[i], corpus[j]) for i, j in enumerate(closest)])


###########################################
# Word Embeddings #########################
###########################################

########################
# Gensim
########################

# Feature Engineering -> text to numeric
list(gensim.downloader.info()['models'].keys())         #
wv = gensim.downloader.load('glove-wiki-gigaword-50')
wv = gensim.downloader.load('word2vec-google-news-300')

# What are word vectors / embeddings?
wv["car"]
wv.similarity("good", "better")
#wv["covid"]
wv.most_similar("car")
wv.most_similar(positive=['king', 'woman'], negative='man')
wv.similar_by_vector(wv['king'] - wv['man'] + wv['woman'])
wv.similar_by_vector(wv['France'] - wv['Paris'] + wv['Berlin'])

vocabulary = set(wv.index_to_key)

# Identify similar documents
# Document Embeddings = Average Word Embeddings

vocabulary = set(wv.index_to_key)

def avg_embeddings(document):
    words = wpt.tokenize(document)
    invocab = [word for word in words if word in vocabulary]        # it is important that the word is in the vocabulary or we'll get error
    avg = np.mean(wv[invocab], axis=0) if len(invocab) >= 1 else []
    return avg


doc_embeddings = [avg_embeddings(doc) for doc in normalized_corpus]
similarity = cosine_similarity(doc_embeddings)
sns.clustermap(similarity, yticklabels=corpus)


########################
# Spacy
# https://spacy.io/models/en/
# With spacy you don't have to calculate the average embeddings yourself
# We can just ask for the similarity of the first document compared to the second document
# In spacy you seem to have a very feature rich environment
########################
nlp = spacy.load("en_core_web_md")
corpus_nlp = [nlp(str(doc)) for doc in normalized_corpus]
corpus_nlp[0]
corpus_nlp[1]
corpus_nlp[0].similarity(corpus_nlp[1])
doc_embeddings = [doc.vector for doc in corpus_nlp]
similarity = cosine_similarity(doc_embeddings)
sns.clustermap(similarity, yticklabels=corpus)


########################
# Visualize Word Embeddings
########################
def pca_plot(model, words):
    vocabulary = model.index_to_key
    word_vectors = np.array([model[word]
                            for word in words if word in vocabulary])
    twodim = PCA().fit_transform(word_vectors)[:, 0:2]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(twodim[:, 0], twodim[:, 1], edgecolors='k', c='r')
    for word, (x, y) in zip(words, twodim):
        plt.text(x+0.05, y+0.05, word)
    return fig


pca_plot(wv,
         ['Berlin', 'Germany', 'Tokyo', 'Japan', 'Athen', 'Greece', 'London', 'UK',
          'France', 'Paris', 'Manila', 'China'])
