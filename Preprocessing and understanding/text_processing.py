import re
import numpy as np
import spacy # often used in industry application, it's fast
import nltk # often used in academic background
from nltk.corpus import gutenberg # this imports some text we're going to look at
from nltk.corpus import wordnet 


#############################
# Regex -> Search/Find/Remove special characters #
# (https://regex101.com/)   #
#############################


text = "ABC def 123 ;:-_!😇" # the smiley is a UTF-8 character. It's perfectly fine to use it.
re.search('ABC', text).span()  # search something in 'text'
re.findall('\d', text)      # Digit [0-9] --- this says "find all instances"
re.findall('\D', text)      # Non-digit [^0-9]
re.findall('\w', text)      # Alphanumeric [A-Za-z0-9_]
re.findall('\W', text)      # Non-alphanumeric [^A-Za-z0-9_]
re.findall('\s', text)
re.findall('\d+', 'Tel: 0431-234234')
re.findall('\d{2}', 'Tel: 0431-234234')
text2 = 'Email: max.mustermann@gmail.com, Mobil: 0151 234123'       # a text
re.search(r'\S+@\w+\.(com|de)', text2)      # search E-Mail addresses
re.findall('\d{4}', '023 1 0431 ')

# Replace
text = "ABC def 123 ;:-_!😇" 
re.sub('\W', '', text)      # searching for non-alphanumerics and if I find it, replace it by "nothing"
re.findall('abc', text)      # 

#############################
# Removing special chars in a function
#############################

def remove_special_characters(text):
    """Remove all characters that are not alphanumeric or space"""
    pattern = r'[\W\s]'
    text = re.sub(pattern, '', text)
    return text


remove_special_characters("ABCDEFG abcdefg;:-.'123#@!😇")


#############################
# Stemming ##################
#############################
text = """The fool does think he is wise, but the wise man knows himself to be a fool."""

text2 = re.sub(",", "", text)
ls = nltk.stem.LancasterStemmer()
' '.join([ls.stem(word) for word in text2.split(" ")])

ls = nltk.stem.PorterStemmer()      # the PorterStemmer seems to be superior!
' '.join([ls.stem(word) for word in text.split(" ")])

###############
# Lemmatizing #
###############

from spacy.lang.en.examples import sentences

# for spacy you need to download packages. This is English. https://spacy.io/models/en/
# click on the right hand side on downloads
# then put into CMD by description python -m spacy download model --direct --sdist pip_args
# with the exact model that oyu want to download

nlp = spacy.load('en_core_web_sm')     
text = nlp(text)
' '.join([word.lemma_ for word in text])

# for words we can extract all sorts of stuff. Here we want to see the lemmas of the word in text
# It is evident that lemmatizing is superior - but it is slower. Which is okay, got the powa :D


#####################
# Nomalize document #
#####################

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

# You can also build a function in order to normalize stuff

def normalize_document(doc):
    doc = re.sub(r'[^a-zA-z0-9\s]', '', doc)    # replacing certain patterns or removing them
    #doc = re.sub('[^a-zA-z0-9\\s]', '', doc)    # without the r we need to use a double dash
    doc = doc.lower()                           # lower casing
    doc = doc.strip()                           # stripping is to remove white cases in the beginning or the end
    tokens = wpt.tokenize(doc)                  # using a Tokenizer I have defined earlier (WordPunctTokenizer)
    filtered_tokens = [token for token in tokens if token not in stop_words] # take the tokens which are not in stop_words
    doc = ' '.join(filtered_tokens)
    doc = ' '.join([word.lemma_ for word in nlp(doc)])
    return doc


texts = ['The fool does think he is wise.',
         'The wise man knows himself to be a fool.']

# Document normalizer
normalize_document(texts[0])    # since it is a list or array or whatever we must select the index
normalize_document(texts)       # texts was a spacy object so we couldn't run it on that so we use: Corpus Normalizer

# Corpus Normalizer
normalize_corpus = np.vectorize(normalize_document)     # this function can normalize the entire list of texts
normalize_corpus(texts)                                 # it gives back a list of cleaned text

#####################
# Test
######################
gutenberg.fileids()
moby = gutenberg.raw('melville-moby_dick.txt')   # Document
moby_corpus = nltk.sent_tokenize(moby)           # Corpus
moby[0:100]
normalize_document(moby[0:100])
normalize_corpus(moby_corpus[0:10])
