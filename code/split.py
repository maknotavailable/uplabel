"""
Split dataset for labeling.


"""
from sklearn.model_selection import StratifiedKFold, train_test_split

def calculate_split(data_length, complexity, labelers):
    #TODO: integrate overlap in split
    #TODO: consider adjusting split size based on #labelers
    #TODO: n_splits is the test/train split. this should be updated to labelers
    count_per_labeler = ((data_length  * (1-complexity)) / labelers)
    if count_per_labeler < 20:
        return 9, 1
    else:
        return 9, 1-complexity

def get_split(data, n_splits=5, max_count=1, idx_label='pred_id'):
    splits = []
    count = 1
    skf = StratifiedKFold(n_splits=max_count, shuffle=True, random_state=123)
    for index, _ in skf.split(data, data[idx_label]):
        splits.append(index)
    #TODO: reduce split sized for high performing categories
    return splits

def get_sample(data, sample_size):
    #TODO: activate learning sort
    #TODO: increase quantity of low available cats
    _data = data.copy()
    return _data.sample(frac=sample_size, random_state=123)

def apply_split(data, fn, complexity, labelers, iter_id, idx_label='pred_id'):
    """Apply split to labeling dataset.

    NOTE: define split complexity by increasing n_splits, and number of splits
    by changing max_count
    """
    _data = data.copy()
    # Determine size based on complexity
    n_splits, sample_size = calculate_split(len(_data), complexity, labelers)

    # Sample based on complexity
    _data = get_sample(_data, sample_size)

    # Apply split
    #TODO: apply active learning to split (a: confidence, b: difference)
    #TODO: if iteration == 0, create test set. else load given test set.
    split_indexes = get_split(_data, n_splits=n_splits, max_count=labelers, idx_label=idx_label)
    df_splits = []
    for count, split in enumerate(split_indexes):
        count += 1
        _temp = _data[_data.index.isin(split)].copy()
        df_splits.append(_temp)
        # Save file
        ## File naming -> [original]-[it_#]-[split_#]
        _fn_temp = '.'.join(fn.split('.')[:-1]) + f'-it_{iter_id}-split_{count}.xlsx'
        print(f'[INFO] Storing split: {_fn_temp}')
        print(f'{_temp.pred.value_counts()}')
        _temp.loc[:,'comment'] = ''
        _temp = _temp[['text','pred','tag','comment']]
        _temp.columns = ['text','label','tag','comment']
        _temp.to_excel(_fn_temp, encoding='utf-8')
    return df_splits