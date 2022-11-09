import pandas as pd
import spacy
from spacy import displacy


##########################
# Spacy                  #
##########################

nlp = spacy.load("en_core_web_sm")    # disable=["tagger", "parser", "ner"]) 

# If I am not interested in ner (named entity recognition) then I lower the performance requirement

doc = nlp("James Bond went to High School in Michigan")
pd.DataFrame([(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
             token.morph, token.ent_type_) for token in doc],
             columns=['text', 'lemma', 'pos', 'tag', 'dep', 'morph', 'ent'])

# So Spacy has been identifying for us several things.
# It shows the word, the lemma, its POS (part of speech), tag (a deeper level of detail, look it up), dep etc.
# There should be a detailed information on the things seen here.
# 'morph' is cool, it shows it's in the Past Tense - it has already happened.
# 'ent' is Entity, it recognizes James Bond in this example ;-) 

# Internally Spacy is storing most of the stuff as numbers. Hence if you don't use the underscore, it will
# store the things as a number instead as a more readible string. If we use ML, we may want to keep numbers.


#############
## POS(part of speech) tags #
#############

# Available POS
[(label, spacy.explain(label)) for label in nlp.get_pipe("tagger").labels]      # Getting a dictionary of what's there

# Extract single POS types
[token for token in doc if token.pos_ == "PROPN"]
[token for token in doc if token.pos_ == "VERB"]
[token for token in doc if token.pos_ == "ADJ"]

#############################
## Named Entity Recognition #
#############################
doc = nlp("Apple is looking at buying U.K. startup DayOff for $1 billion")

# Document-level
[(ent.text, ent.label_) for ent in doc.ents]
displacy.render(doc, style="ent")

# Token-level
[(token.text, token.ent_type_) for token in doc if token.ent_type_ != ""]
spacy.explain('GPE')


#######################
## Linguistic parsing #
#######################

# Shallow parsing
doc = nlp("Autonomous cars shift insurance liability toward manufacturers")
pd.DataFrame([(chunk.text, chunk.root.text)
             for chunk in doc.noun_chunks], columns=['noun phrase', 'noun'])

# Dependency parsing
doc = nlp('Satellites spot whales from space')
options = {'compact': True, 'add_lemma': False, 'distance': 150}
displacy.render(doc, style="dep", options=options)   #
pd.DataFrame([(token.text, token.dep_, spacy.explain(token.dep_), token.head.text)
             for token in doc], columns=['text', 'dep1', 'dep2', 'headtext'])
