"""
Estimate complexity

"""
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from nltk.corpus import stopwords  

from sklearn.naive_bayes import BernoulliNB, MultinomialNB, ComplementNB

import pandas as pd
import numpy as np

import pickle

import utils as ut

def get_map_labels(labels):
    return labels.reset_index().drop('label', axis=1).reset_index().set_index('index').T.to_dict('int')['level_0']

def check_labels(data, min=10, mid=50):
    out = dict(high = [], medium = [], low = [])
    labels = data.label.value_counts()
    for lbl, idx in zip(labels, labels.index):
        if lbl < min:
            print(f'[INFO] Not enough examples for label {idx, lbl}')
            out['low'].append(idx)
        elif lbl < mid:
            # print(f'[INFO] Not enough examples for label {idx, lbl}')
            out['medium'].append(idx)
        else:
            out['high'].append(idx)
    return out, get_map_labels(labels)

def prepare_split(data, split=True, vectorize=False, oversample=False):
    # Clean data
    data['training_clean'] = ut.prepare_text(data)
    if split:
        # Split
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        for train_index, test_index in skf.split(data.training_clean, data.cat_id):
            # Content
            X_train, X_test = data.training_clean[train_index], data.training_clean[test_index]
            # Labels
            y_train, y_test = data.cat_id[train_index], data.cat_id[test_index]
    else:
        X_train = data.training_clean
        X_test = data.training_clean
        y_train = data.cat_id
        y_test = data.cat_id

    if vectorize:
        # X_train = get_vector(X_train)
        # X_test = get_vector(X_test)
        pass

    # Optional: oversample minority class
    if oversample:
        smote = SMOTE('minority', random_state=12)
        X_train, y_train = smote.fit_sample(X_train, y_train)
    else:
        y_train.plot('hist')
        y_test.plot('hist')
    return X_train, y_train, X_test, y_test

def get_cat_complexity(data, cat_map, vectorize=False, oversample=False, save=False):
    _data = data.copy()
    #TODO: fixate test split for use case

    # Assign labels
    _data['cat_id'] = _data['label'].apply(lambda x: ut.apply_cat_id(x, cat_map))
    _data = _data[_data['cat_id'] != -1].copy()
    print(f'[INFO] Data available for training -> {len(_data)}')

    # Clean data & Split data
    X_train, y_train, X_test, y_test = prepare_split(_data, vectorize=vectorize, oversample=oversample)

    # Train
    ## Build Pipeline
    #TODO: integrate preprocessor=ut.clean_text,
    text_clf = Pipeline([
        ('vect', HashingVectorizer(alternate_sign = False, stop_words = stopwords.words('german'), n_features = 2**16)),
        ('tfidf', TfidfTransformer()),
        ('clf', ComplementNB()),
    ])
    ## Train
    text_clf.fit(X_train, y_train)

    # Store
    if save:
        out_file = '../model/cat.ml'
        with open(out_file, "wb") as fn:
            pickle.dump(text_clf, fn)
            pickle.dump(cat_map, fn)

    ## Test
    pred = text_clf.predict(X_test)
    print('[INFO] Complexity Estimation Report: \n',classification_report(y_test, pred))  

    # Calculate Score
    complexity = np.mean(pred == y_test)
    return complexity, text_clf

def apply_cat(data, cat_model, cat_map=None):
    text = ut.prepare_text(data)
    data['pred_id'] = cat_model.predict(text)

    if cat_map is not None:
        data['pred'] = data.pred_id.apply(lambda y: [list(cat_map.keys())[list(cat_map.values()).index(x)] for x in [y]][0])

    return data

def run(data, estimate_clusters):
    """Run function for complexity task

    INPUT
    - data (dataframe)

    OUTPUT
    - score (float) : complexity score
    - report (dataframe)
    - model
    """
    _data = data.copy()

    # Step 1 - check labels
    res_labels, map_labels = check_labels(_data)

    # Step 2 
    if len(res_labels['low']) > 0 and estimate_clusters:
        print('[INFO] Clustering needed for split.')
        complexity, model = None, None
    else:
        print('\n[INFO] Estimating complexity.')
        complexity, model = get_cat_complexity(_data, map_labels)
    
    # Step 3
    if model is not None:
        print('\n[INFO] Applying model to data')
        data_tagged = apply_cat(_data, model, map_labels)

    return complexity, model, data_tagged