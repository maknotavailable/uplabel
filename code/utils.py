"""
Util functions for data preparation and formatting.


"""
import pandas as pd
import re
import spacy
from sklearn.model_selection import StratifiedKFold

def clean_text(text, nlp):
    """Basic text cleaning for training and evaluation.

    INPUT
        text (string)
    OUTPUT
        text (string) : cleaned
    """
    if type(text) != str:
        # Error catching for non-strings
        print('\t[INFO] Found an empty row during text cleaning.')
        text = ' '
    else:
        # Lemmatize
        doc = nlp(text)
        text = ' '.join([t.lemma_ for t in doc])

    # Remove all the special characters
    text = re.sub(r'\W', ' ', text)

    # Remove all single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

    # Remove single characters from the start
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) 
    
    # Remove all numbers
    text = re.sub(r'[0-9]', ' ', text)
    
    # Substituting multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    return text.lower()

def prepare_text(data, language='de', col='text'):
    nlp = spacy.load(language)
    return data['text'].apply(lambda x: clean_text(x, nlp))


def apply_cat_id(x, labels):
    """Map Categories to Numeric Labels"""
    try:
        return int(labels[x])
    except:
        return -1

def append_iter():
    """Add iteration ID to DataFrame"""
    #TODO:
    pass