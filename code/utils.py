"""
Util functions for data preparation and formatting.


"""
import pandas as pd
import spacy
from sklearn.base import TransformerMixin 

class CleanText(TransformerMixin):
    """Basic text cleaning for model training and scoring."""

    def __init__(self, language='de', stopwords=None):
        self.nlp = spacy.load(language, disable=['ner','parser','tagger'])
        if stopwords is not isinstance(stopwords, list):
            try:
                with open('../assets/stopwords_' + language +'.txt', encoding='utf-8') as fn:
                    self.stopwords = fn.read()
            except:
                print(f'[WARNING] No stopword list found for {language}.')
                self.stopwords = []

    def clean_text(self, text):
        #TODO: create custom spacy pipeline (docs = list(nlp.pipe(df.text)))
        if type(text) != str:
            print('\t[INFO] Found an empty row during text cleaning.')
            text = ' '
        else:      
            # Lemmatize and remove stopwords
            text = ' '.join([t.lemma_ for t in self.nlp(text) if t.text not in self.stopwords])   
        return text.strip().lower()

    def transform(self, texts):
        df_texts = pd.Series(texts)
        df_texts = df_texts.apply(self.clean_text)
        return df_texts.values

    def fit(self, *_):
        return self

# def prepare_text(data, language='de', col='text', stopwords=None):
#     nlp = spacy.load(language, disable=['ner','parser','tagger'])
#     if stopwords is not None and not isinstance(stopwords, list):
#         with open('../assets/stopwords_' + language +'.txt', encoding='utf-8') as fn:
#             stopwords = fn.read()
#     if isinstance(data, str):
#         res = clean_text(data, nlp, stopwords)
#     else:
#         res = data[col].apply(lambda x: clean_text(x, nlp, stopwords))
#     return res

def apply_pred_id(x, labels):
    """Map Categories to Numeric Labels"""
    try:
        return int(labels[x])
    except:
        return -1