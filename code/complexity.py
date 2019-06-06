"""
Estimate task complexity

"""
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfTransformer, HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, silhouette_score, f1_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import utils as ut
from utils import CleanText

def get_map_labels(labels):
    return labels.reset_index().drop('label', axis=1).reset_index().set_index('index').T.to_dict('int')['level_0']

def check_labels(data, min=10, mid=50):
    out = dict(high = [], medium = [], low = [])
    labels = data[data.label != ''].label.value_counts()
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
    if split:
        # Split
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=222)
        for train_index, test_index in skf.split(data.text, data.cat_id):
            # Content
            X_train, X_test = data.text[train_index], data.text[test_index]
            # Labels
            y_train, y_test = data.cat_id[train_index], data.cat_id[test_index]
    else:
        X_train = data.text
        X_test = data.text
        y_train = data.cat_id
        y_test = data.cat_id

    return X_train, y_train, X_test, y_test

def get_cat_complexity(data, cat_map, language, vectorize=False, oversample=False, save=False):
    _data = data.copy()
    #TODO: fixed test split per case/project

    # Assign labels
    _data['cat_id'] = _data['label'].apply(lambda x: ut.apply_pred_id(x, cat_map))
    _data = _data[_data['cat_id'] != -1].reset_index(drop=True).copy()
    print(f'\t[INFO] Data available for training -> {len(_data)}')

    # Create train / test split
    X_train, y_train, X_test, y_test = prepare_split(_data, vectorize=vectorize, oversample=oversample)

    ## Build Pipeline   
    pipe = Pipeline([
        ('clean', CleanText(language=language)),
        ('vect', HashingVectorizer(alternate_sign = False, n_features = 2**16)),
        ('tfidf', TfidfTransformer()),
        ('smt', SMOTE(ratio='all',random_state=42)),
        # ('clt', ComplementNB())
        ('clf', SGDClassifier(loss='modified_huber', max_iter=1000, tol=1e-3, random_state=123)),
    ])
    ## Train
    pipe.fit(X_train, y_train)

    # # Store
    # #TODO: store model for scoring application
    # if save:
    #     out_file = '../model/cat.ml'
    #     with open(out_file, "wb") as fn:
    #         pickle.dump(pipe, fn)
    #         pickle.dump(cat_map, fn)

    ## Test
    pred = pipe.predict(X_test)
    print('\t[INFO] Complexity Estimation Report: \n',classification_report(y_test, pred))  

    # Calculate Score
    complexity = f1_score(y_test, pred, average='weighted')
    ## TODO: add data used for training -> (len(_data) / len(data)))*100
    ## TODO: normalize complexity (softmax?)
    return complexity, pipe

def get_cluster_complexity(data, n_cluster, language):
    # Fit cluster model
    pipe = Pipeline([
        ('clean', CleanText(language='de')),
        ('vect', TfidfVectorizer(max_df=0.5, min_df=2)),
        ('tfidf', TfidfTransformer()),
        ('clf', KMeans(n_clusters=n_cluster, init='k-means++', max_iter=100, n_init=1, random_state=222)),
    ])
    pipe.fit(data.text)
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=2)
    X = vectorizer.fit_transform(CleanText(language='de').transform(data.text))

    # Calculate complexity
    pred = pipe.predict(data.text)
    silscore = silhouette_score(X, pred)
    complexity = (silscore + 1)/2
    
    # Mapping Table
    _temp = data.copy()
    print(data.label.value_counts())
    _temp['cluster'] = pred #TODO: getting mapping table is broken
    _temp = _temp[["index","cluster", "label"]].groupby(["cluster", "label"]).count().sort_values("index").groupby(level=0).tail(1)
    _temp = _temp.reset_index()
    cat_map = _temp.groupby('cluster')['label'].apply(lambda x: x).to_dict()

    return complexity, pipe, cat_map

def get_al_score(data, cat_model):
    """Calculate Active Learning Score used in Split
    
    Method: Margin Sampling
    """
    _proba = cat_model.predict_proba(data.text)
    _sorted = np.sort(_proba, axis = 1)
    return np.subtract(_sorted[:,_proba.shape[1]-1],_sorted[:,_proba.shape[1]-2])

def apply_pred(data, cat_model, cat_map, unsupervised):
    """Suggest labels """
    data['pred_id'] = cat_model.predict(data.text)
    if unsupervised:
        data['pred'] = data.pred_id.apply(lambda y: cat_map[y])
    else:
        data['pred'] = data.pred_id.apply(lambda y: [list(cat_map.keys())[list(cat_map.values()).index(x)] for x in [y]][0])
        
    return data

def run(data, estimate_clusters, language):
    """Run function for complexity task

    INPUT
    - data (dataframe)

    OUTPUT
    - score (float) : complexity score
    - model (object) : sklearn pipeline
    - report (dataframe)
    """
    _data = data.copy()
    unsupervised = False

    # Step 1 - check labels
    res_labels, map_labels = check_labels(_data)

    # Step 2 - calculate complexity
    if len(res_labels['low']) > 0 and estimate_clusters:
        print('\n[INFO] Estimating complexity using UNSUPERVISED approach.')
        complexity, model, map_labels = get_cluster_complexity(_data, len(map_labels), language)
        unsupervised = True
    else:
        print('\n[INFO] Estimating complexity using SUPERVISED approach.')
        complexity, model = get_cat_complexity(_data, map_labels, language)
    print(f'\t[INFO] Complexity Score -> {complexity:.3}')
    
    # Step 3
    print('\t[INFO] Applying model to data')
    data_tagged = apply_pred(_data, model, map_labels, unsupervised=unsupervised)
    if not unsupervised:
        data_tagged['al_score'] = get_al_score(data_tagged, model)
    else: 
        data_tagged['al_score'] = 0

    return complexity, model, data_tagged