from sklearn import metrics
import numpy as np
import gensim
import nltk
import re
from gensim import downloader


stop_words = nltk.corpus.stopwords.words('english')
wpt = nltk.WordPunctTokenizer()


def normalize_document(doc):
    """Normalize the document (lower case, stopword removal, ...)"""

    doc = re.sub(r'@[\w]+', '', doc)          # replace user mentions
    doc = re.sub(r'http[\S]+', 'URL', doc)    # replace URLs
    doc = re.sub(r'[^\w\s]', '', doc)         # keep words and spaces
    doc = doc.lower()
    doc = doc.strip()
    tokens = wpt.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens)
    return doc


def evaluate(true, pred):
    """Evluation metrics for classification"""

    ac = metrics.accuracy_score(true, pred)
    pr = metrics.precision_score(true, pred)
    re = metrics.recall_score(true, pred)
    f1 = metrics.recall_score(true, pred)
    return {'accuracy': ac, 'precision': pr, 'recall': re, 'f1-score': f1}


def avg_embeddings(words, model, vocabulary):

    num_features = model.vector_size
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0.

    for word in words:
        if word in vocabulary:
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model[word])

    if nwords:
        feature_vector = np.divide(feature_vector, nwords)

    return feature_vector


def avg_embeddings_corpus(corpus, model):
    vocabulary = set(model.index_to_key)
    features = [avg_embeddings(tokenized_sentence, model, vocabulary)
                for tokenized_sentence in corpus]
    return np.array(features)
