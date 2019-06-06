"""
Split dataset for labeling.


"""
from sklearn.model_selection import StratifiedKFold

def get_split_size(data_length, complexity, labelers):
    #TODO: consider adjusting split size based on #labelers
    #TODO: n_splits is the test/train split. this should be updated to labelers
    count_per_labeler = ((data_length  * (1-complexity)) / labelers)
    if count_per_labeler < 20:
        return 9, 1
    else:
        return 9, 1-complexity

def get_sample(data, sample_size):
    # Sort based on active learning score
    al_split = int(len(data) * sample_size)
    _data = data[data.label == ''].sort_values(by='al_score').head(al_split).copy()
    ## Include quality control data
    quality_size = int(0.1 * len(data[data.label != ''])) #TODO: get from params
    print(f'\t[INFO] Quality Size -> {quality_size}')
    _data_quality = data[data.label != ''].sort_values(by='al_score').head(quality_size).copy()
    _data = _data.append(_data_quality, sort=False, ignore_index=True)
    _data.drop('al_score', axis=1, inplace=True)

    # Increase quantity of low represented categories
    #TODO: increase quantity of low available cats
    return _data.sample(frac=1, random_state=222)

def get_split(data, n_splits=5, max_count=1, idx_label='pred_id'):
    splits = []
    count = 1
    skf = StratifiedKFold(n_splits=max_count, shuffle=True, random_state=123)
    for index, _ in skf.split(data, data[idx_label]):
        splits.append(index)
    return splits

def get_best_label(df):
    if df['label'] == '':
        return df['pred']
    else:
        return df['label']

def apply_split(data, fn, complexity, labelers, iter_id, max_split, idx_label='pred_id'):
    """Apply split to labeling dataset.

    Output splits are named as follows:
        [original fn]-[it_#]-[split_#]

    NOTE: define split complexity by increasing n_splits, and number of splits
    by changing max_count
    """
    _data = data.copy()
    # Determine size based on complexity
    n_splits, sample_size = get_split_size(len(_data), complexity, labelers)
    if max_split is not False:
        if (sample_size * len(_data)) / labelers > max_split:
            sample_size = (max_split * labelers) / len(data) 
    print(f'\n[INFO] Applying split \n\t[INFO] Number of splits -> {labelers} \n\t[INFO] Sample Fraction -> {sample_size:.4%}')
    # Sample based on complexity
    _data = get_sample(_data, sample_size)

    # Create overlap for quality score
    _data_quality = _data[_data.label != ''].copy()
    if len(_data_quality) > 50:
        _data_quality = _data_quality.sample(frac=1, random_state=222).head(30)
    _data = _data[_data.label == ''] 

    # Create overlap for consistancy score
    overlap_size = int((0.1 * len(_data)) / labelers) #TODO: get from params
    if overlap_size > 50:
        overlap_size = 50
    _data_overlap = _data.head(overlap_size).copy()
    _data = _data.tail(len(_data)-overlap_size)

    print(f'\t[INFO] Overlap Size -> {overlap_size}')
    #TODO: log

    # Apply split
    #TODO: if iteration == 0, create test set. else load given test set.
    #TODO: Visualize split
    _data.reset_index(drop=True, inplace=True)
    if labelers == 1:
        df_splits = []
        ## Append overlap
        _data = _data.append(_data_overlap, sort=False, ignore_index=True)
        ## Append quality
        _data = _data.append(_data_quality, sort=False, ignore_index=True)
        _data['label_out'] = _data.apply(get_best_label, axis = 1) 
        _data = _data.sample(frac=1, random_state=222)
        _data.set_index('index', inplace=True)
        df_splits.append(_data)

        # Save file
        _fn_temp = '.'.join(fn.split('.')[:-1]) + f'-it_{iter_id}-split_{labelers}.xlsx'
        _data = _data[['text','label_out','tag','comment']]
        _data.columns = ['text','label','tag','comment']
        _data.to_excel(_fn_temp, encoding='utf-8')

        print(f'[INFO] Created split: {_fn_temp}')
        print(f'{_data.label.value_counts()}')

    else:
        split_indexes = get_split(_data, n_splits=n_splits, max_count=labelers, idx_label=idx_label)
        df_splits = []
        for count, split in enumerate(split_indexes):
            count += 1

            # Fetch Split
            _temp = _data.iloc[split].copy()
            ## Append overlap
            _temp = _temp.append(_data_overlap, sort=False, ignore_index=True)
            ## Append quality
            _temp = _temp.append(_data_quality, sort=False, ignore_index=True)
            _temp['label_out'] = _temp.apply(get_best_label, axis = 1) 
            _temp = _temp.sample(frac=1, random_state=222)
            _temp.set_index('index', inplace=True)
            df_splits.append(_temp)

            # Save file
            _fn_temp = '.'.join(fn.split('.')[:-1]) + f'-it_{iter_id}-split_{count}.xlsx'
            _temp = _temp[['text','label_out','tag','comment']]
            _temp.columns = ['text','label','tag','comment']
            _temp.to_excel(_fn_temp, encoding='utf-8')

            print(f'[INFO] Created split: {_fn_temp}')
            print(f'{_temp.label.value_counts()}')
    return df_splits