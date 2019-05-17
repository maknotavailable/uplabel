"""
Split dataset for labeling.


"""
from sklearn.model_selection import StratifiedKFold

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
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
    for index, _ in skf.split(data, data[idx_label]):
        splits.append(index)
        if count == max_count:
            break
        count += 1   
    #TODO: reduce split sized for high performing categories
    return splits

def get_sample(data, sample_size):
    _data = data.copy()
    return _data.sample(frac=sample_size, random_state=123)

def apply_split(data, fn, complexity, labelers, idx_label='pred_id'):
    """Apply split to labeling dataset.

    NOTE: define split complexity by increasing n_splits, and number of splits
    by changing max_count
    """
    # Determine size based on complexity
    n_splits, sample_size = calculate_split(len(data), complexity, labelers)

    # Sample based on complexity
    data = get_sample(data, sample_size)

    # Apply split
    #TODO: apply active learning to split (a: confidence, b: difference)
    #TODO: if iteration == 0, create test set. else load given test set.
    split_indexes = get_split(data, n_splits=n_splits, max_count=labelers, idx_label=idx_label)
    df_splits = []
    for count, split in enumerate(split_indexes):
        count += 1
        _temp = data[data.index.isin(split)]
        df_splits.append(_temp)
        # Save file
        _fn_temp = '.'.join(fn.split('.')[:-1]) + '_split' + str(count) + '.xlsx'
        print(f'[INFO] Storing split: {_fn_temp}')
        print(f'{_temp.pred.value_counts()}')
        _temp.loc[:,'comment'] = ''
        _temp[['iter_id','text','pred','label','tag','comment']].to_excel(_fn_temp, encoding='utf-8')

    return df_splits