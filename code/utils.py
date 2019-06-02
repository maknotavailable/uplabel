"""
Util functions for data preparation and formatting.


"""
import pandas as pd
import re
import spacy

def clean_text(text, nlp, stopwords):
    """Basic text cleaning for training and evaluation.

    INPUT
        text (string)
    OUTPUT
        text (string) : cleaned

    #TODO: create custom spacy pipeline (docs = list(nlp.pipe(df.text)))
    """
    if stopwords is None:
        stopwords = []
    if type(text) != str:
        # Error catching for non-strings
        print('\t[INFO] Found an empty row during text cleaning.')
        text = ' '
    else:      
        # Lemmatize
        text = ' '.join([t.lemma_ for t in nlp(text) if t.text not in stopwords])   
    return text.strip().lower()

def prepare_text(data, language='de', col='text', stopwords=None):
    nlp = spacy.load(language, disable=['ner','parser','tagger'])
    if stopwords is not None and not isinstance(stopwords, list):
        with open('../assets/stopwords_' + language +'.txt', encoding='utf-8') as fn:
            stopwords = fn.read()
    if isinstance(data, str):
        res = clean_text(data, nlp, stopwords)
    else:
        res = data[col].apply(lambda x: clean_text(x, nlp, stopwords))
    return res

def apply_pred_id(x, labels):
    """Map Categories to Numeric Labels"""
    try:
        return int(labels[x])
    except:
        return -1