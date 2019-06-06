import os
import pandas as pd
import numpy as np

def assign_lbl_score(x):
    if x == '':
        return 0
    else:
        return 1

def prepare_data(data):
    return data.replace(np.nan, '', regex=True)

def load_splits(data_dir, iter_id):
    # Load Files
    df_splits = []
    try:
        for _fn in os.scandir(data_dir):
            if len(_fn.name.split('.')[0].split('-')) > 2:
                _split = _fn.name.split('.')[0].split('-')[-1].split('_')[-1]
                _iteration = int(_fn.name.split('.')[0].split('-')[-2].split('_')[-1])
                if 'residual' == _split and iter_id == _iteration:
                    residual = pd.read_csv(_fn.path, sep='\t', encoding='utf-8', index_col=0)
                    residual = prepare_data(residual)
                    path = '-'.join(_fn.name.split('-')[:2]) + '_train.txt'
                    residual['lbl_score'] = residual['label'].apply(assign_lbl_score)
                elif iter_id == _iteration:
                    _temp_split = pd.read_excel(_fn.path, encoding='utf-8', index_col=0)
                    _temp_split = prepare_data(_temp_split)
                    _temp_split = _temp_split.reset_index(drop=False)
                    _temp_split['labeler'] = _split
                    df_splits.append(_temp_split)
    except Exception as e:
        print(f'[ERROR] Files seem to be missing, or you may have skipped an iteration -> {e}') 
    return residual, df_splits, path

def get_quality_score(df_truth, df_split):
    """Overlap with ground truth"""

    _available = df_truth[df_truth.label != ''].merge(df_split, on = ['index']).copy()
    _correct = len(_available[(_available['label_x'] == _available['label_y'])])
    score = _correct / len(_available)
    if len(_available) == 0:
        print(f'\t[INFO] No quality overlap detected.')
    return score

def get_consistance_score(df_split):
    """Overlap with other labelers"""

    # Group by Labeler
    grouped_labelers = df_split.groupby('labeler')
    grouped_labels = df_split.groupby('index')
    
    scores = []
    major = []

    # Majority Vote: removed if no distinct majority
    for name, group in grouped_labels:
        gsum = group.label.mode()
        if len(gsum) == 1:
            major.append({'index': name, 'label': gsum[0]})
    major = pd.DataFrame(major)

    # OVERLAP WITH MAJORITY
    scores = dict()
    for name, group in grouped_labelers:
        overlap_major = group.merge(major, on=['index', 'label'])
        gt_overlap = len(overlap_major) / len(major)
        scores[name] =  gt_overlap
    return scores

def get_lbl_score(_residual, _df_splits_list):
    print("\t[INFO] Calculating labeler score")
    
    _df_splits = pd.concat(_df_splits_list, ignore_index = True, sort = False)
    consistance_score = get_consistance_score(_df_splits)
    _df_splits_list_out = []
    for _split in _df_splits_list:
        labeler = _split['labeler'].drop_duplicates().values[0]
        quality_score = get_quality_score(_residual, _split)
        labeler_score = (quality_score*2 + consistance_score[labeler])/3
        _split['lbl_score'] = labeler_score
        _df_splits_list_out.append(_split)
        print(f'\t[INFO] Labeler {labeler} Score -> {labeler_score:.2}')
    return _df_splits_list_out

def join_splits(residual, df_splits):
    _data = residual.append(df_splits, sort=False, ignore_index=True) 
    _data.sort_values(by=['lbl_score'], ascending=False, inplace=True)
    _data.drop_duplicates(subset=['text'], keep='first', inplace=True)
    _data.sort_values(by=['index'], ascending=True, inplace=True)
    _data.set_index('index', inplace=True)
    _data.drop(['lbl_score','al_score'], axis=1, inplace=True)
    return _data

def load_iteration(data_dir, iter_id):
    # Load
    residual, df_splits, path = load_splits(data_dir, iter_id)
    # Get Labeler Scores
    df_splits = get_lbl_score(residual, df_splits)
    # Merge
    data = join_splits(residual, df_splits)
    # Store
    data.to_csv(data_dir + path, sep='\t', encoding='utf-8')

    return data

