import os
import pandas as pd
import numpy as np

# class IterationControl:

def get_quality_score(df_truth, df_split):
    """Overlap with ground truth"""
    _available = df_truth.merge(df_split, on=['index']).copy() #right_index=True, left_index=True
    _correct = _available[(_available['label_x'] == _available['label_y'])].copy()
    score = len(_correct) / len(_available)
    return score

def get_consistance_score(df_split):
    """Overlap with other labelers"""
        
    #TODO:
    pass

def load_splits(data_dir, iter_id):
    # Load Files
    df_splits = []
    for _fn in os.scandir(data_dir):
        if len(_fn.name.split('.')[0].split('-')) > 2:
            _split = _fn.name.split('.')[0].split('-')[-1].split('_')[-1]
            _iteration = int(_fn.name.split('.')[0].split('-')[-2].split('_')[-1])
            
            if 'residual' == _split and iter_id == _iteration:
                residual = pd.read_csv(_fn.path, sep='\t', encoding='utf-8', index_col=0)
                residual.reset_index(drop=False, inplace=True)
                residual.replace(np.nan, '', regex=True, inplace=True)
                path = '-'.join(_fn.name.split('-')[:2]) + '_train.txt'
                def assign_lbl_score(x):
                    if x == '':
                        return 0
                    else:
                        return 1
                residual['lbl_score'] = residual['label'].apply(assign_lbl_score)
            elif iter_id == _iteration:
                _temp_split = pd.read_excel(_fn.path, encoding='utf-8', index_col=0)
                _temp_split.reset_index(drop=False, inplace=True)
                _temp_split.replace(np.nan, '', regex=True, inplace=True)
                quality_score = get_quality_score(residual, _temp_split)
                _temp_split['lbl_score'] = quality_score
                df_splits.append(_temp_split)
                print(f'\t[INFO] Quality Score of Labeler {_split} -> {quality_score}')

    return residual, df_splits, path

def join_splits(residual, df_splits):
    data = residual.append(df_splits, sort=False, ignore_index=True) 
    data.sort_values(by=['lbl_score'], ascending=False, inplace=True)
    data.drop_duplicates(subset=['text'], keep='first', inplace=True)
    data.set_index('index', inplace=True)
    data.drop(['lbl_score'], axis=1, inplace=True)
    return data

def load_iteration(data_dir, iter_id):
    # Load
    residual, df_splits, path = load_splits(data_dir, iter_id)
    # Merge
    data = join_splits(residual, df_splits)
    # Store
    data.to_csv(data_dir + path, sep='\t', encoding='utf-8')

    return data

