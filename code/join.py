import os
import pandas as pd

# class IterationControl:

def get_quality_score():
    """Overlap with ground truth"""
    #TODO:
    pass

def get_consistance_score():
    """Overlap with other labelers"""
    #TODO:
    pass


def join_splits(residual, df_splits):
    #split into extra and required columns
    data = residual.append(df_splits, sort=False, ignore_index=True) #TODO: sort by score
    return data

def load_splits(data_dir, iter_id):
    # Load Files
    df_splits = []
    for _fn in os.scandir(data_dir):
        if len(_fn.name.split('.')[0].split('-')) > 2:
            _split = _fn.name.split('.')[0].split('-')[-1].split('_')[-1]
            _iteration = int(_fn.name.split('.')[0].split('-')[-2].split('_')[-1])
            
            if 'residual' == _split and iter_id == _iteration:
                residual = pd.read_csv(_fn.path, sep='\t', encoding='utf-8')
                path = '-'.join(_fn.name.split('-')[:2]) + '.txt'
            elif iter_id == _iteration:
                _temp_split = pd.read_excel(_fn.path, encoding='utf-8')
                _temp_split['labeler_id'] = _split
                df_splits.append(_temp_split)

    return residual, df_splits, path

def load_iteration(data_dir, iter_id):
    # Load
    residual, df_splits, path = load_splits(data_dir, iter_id)
    # Merge
    data = join_splits(residual, df_splits)
    # Store
    data.to_csv(data_dir + path, sep='\t', encoding='utf-8')

    return data

