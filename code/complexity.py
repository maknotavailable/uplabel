"""
Estimate task complexity

"""
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, ComplementNB
from sklearn.metrics import classification_report, silhouette_score, f1_score
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from utils import CleanText
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
    if split:
        # Split
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        for train_index, test_index in skf.split(data.text_clean, data.cat_id):
            # Content
            X_train, X_test = data.text_clean[train_index], data.text_clean[test_index]
            # Labels
            y_train, y_test = data.cat_id[train_index], data.cat_id[test_index]
    else:
        X_train = data.text_clean
        X_test = data.text_clean
        y_train = data.cat_id
        y_test = data.cat_id

    return X_train, y_train, X_test, y_test

def get_cat_complexity(data, cat_map, vectorize=False, oversample=False, save=False):
    _data = data.copy()
    #TODO: investigate fixed test split for use case

    # Assign labels
    _data['cat_id'] = _data['label'].apply(lambda x: ut.apply_pred_id(x, cat_map))
    _data = _data[_data['cat_id'] != -1].reset_index(drop=True).copy()
    print(f'\t[INFO] Data available for training -> {len(_data)}')

    # Clean data & Split data
    X_train, y_train, X_test, y_test = prepare_split(_data, vectorize=vectorize, oversample=oversample)

    # Train
    ## Build Pipeline
    pipe = Pipeline([
        # ('clean', CleanText(language='de')),TODO:
        ('vect', HashingVectorizer(alternate_sign = False, n_features = 2**16)),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='modified_huber')),
    ])
    ## Train
    pipe.fit(X_train, y_train)

    # Store
    #TODO: store model for scoring application
    if save:
        out_file = '../model/cat.ml'
        with open(out_file, "wb") as fn:
            pickle.dump(pipe, fn)
            pickle.dump(cat_map, fn)

    ## Test
    pred = pipe.predict(X_test)
    print('\t[INFO] Complexity Estimation Report: \n',classification_report(y_test, pred))  

    # Calculate Score
    complexity = f1_score(y_test, pred, average='weighted')
    ## TODO: add data used for training -> (len(_data) / len(data)))*100
    ## TODO: normalize complexity (softmax?)
    print(f'\t[INFO] Complexity Score -> {complexity}')
    return complexity, pipe

def get_cluster_complexity(data, language):
    
    # Fit cluster model
    ## Cluster Count
    nCl = len(data[(~data.label.isna()) & (data.label != '')].label.drop_duplicates()) 
    pipe = Pipeline([
        ('vect', TfidfVectorizer(max_df=0.5, min_df=2)),
        ('tfidf', TfidfTransformer()),
        ('clf', KMeans(n_clusters=nCl, init='k-means++', max_iter=100, n_init=1, random_state=0)),
    ])
    pipe.fit(data.text_clean)
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=2)
    X = vectorizer.fit_transform(data.text_clean)

    # Calculate complexity
    pred = pipe.predict(data.text_clean)
    silscore = silhouette_score(X, pred)
    complexity = (silscore + 1)/2
    print(f'\t[INFO] Complexity Score -> {complexity}')
    
    # Mapping Table
    _temp = data[data.label != ''].copy()
    _temp['cluster'] = pred
    _temp = _temp[["cluster", "label","text"]].groupby(["cluster", "label"]).count().sort_values("text").groupby(level=0).tail(1)
    _temp = _temp.reset_index()
    cat_map = _temp.groupby('cluster')['label'].apply(lambda x: x).to_dict()

    return complexity, pipe, cat_map

def calculate_al_score(proba):
    """
    Calculate Active Learning Score used in Split
    
    Method: Margin Sampling
    """
    # _sorted = proba[np.argsort(proba)]
    # #TODO:
    # print(_sorted)
    # print(_sorted[:,0])
    # _al = np.subtract(_sorted[:,0],_sorted[:,1])
    # print(_al)
    # print(_al.shape)
    return 0

def apply_pred(data, cat_model, cat_map, unsupervised):
    text = ut.prepare_text(data)
    data['pred_id'] = cat_model.predict(text)
    _predict_proba = cat_model.predict_proba(text)
    if unsupervised:
        data['pred'] = data.pred_id.apply(lambda y: cat_map[y])
        data['al'] = None
    else:
        data['pred'] = data.pred_id.apply(lambda y: [list(cat_map.keys())[list(cat_map.values()).index(x)] for x in [y]][0])
        data['al'] = calculate_al_score(_predict_proba)
        
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
    ## Clean data
    _data['text_clean'] = ut.prepare_text(_data)
    if len(res_labels['low']) > 0 and estimate_clusters:
        print('[INFO] Estimating complexity using UNSUPERVISED approach.')
        complexity, model, map_labels = get_cluster_complexity(_data, language)
        unsupervised = True
    else:
        print('\n[INFO] Estimating complexity using SUPERVISED approach.')
        complexity, model = get_cat_complexity(_data, map_labels)
    
    # Step 3
    print('\n[INFO] Applying model to data')
    data_tagged = apply_pred(_data, model, map_labels, unsupervised=unsupervised)

    return complexity, model, data_tagged