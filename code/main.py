"""
Main function to run UpLabel.


"""
import pandas as pd

# Custom functions
import utils as ut
import complexity as cp
import split as sp

def create_instructions():
    #TODO: automatically create instructions
    #WHAT LABELS
    #EXAMPLES PER LABEL
    #WHAT TO EDIT
    pass

def load_input(data, cols, extras, target='label', 
                    language='de', task='cat', labelers=1,
                    quality=1, estimate_clusters=True,
                    iteration = 0):
    """First contact data loading

    INPUT
    - data (dataframe(s) OR string) : dataframe(s) or path to data
    - cols (list)
    - extras (list)
    - target (string)
    - language (string)
    - task (string) :
        cat = classification
        ent = entitiy (NER)
    - labelers (int) 
    - quality (int) : 
        1 = strict
        2 = medium / smart
        3 = ignore

    OUTPUT
    - df_all (dataframe) : entire dataset, includiong all metadata
    - df_split (dataframe(s)) : data split for labeling
    
    #TODO: implement iterations
    """
    if isinstance(data, str):
        _fn = data
        data = pd.read_csv(data, sep='\t', encoding='utf-8')
    
    #check if data is list of DFs or single
    if isinstance(data, (list)):
        print('[TODO] LIST OF DATAFRAMES')
        #TODO: merge DFs

    # PREPARE
    df_split = data[cols].copy()
    df_all = data.copy()
    print(f'[INFO] Input Length -> {len(df_split)}')
    print(f'[INFO] Label Counts: \n{df_split.label.value_counts()}')
    ## Standardize
    df_split.columns = ['text','label','tag']
    ## Drop Duplicates
    df_split.sort_values(by=['label'], inplace=True)
    df_split.drop_duplicates(subset=['text'], inplace=True)
    df_all = df_all[df_all.index.isin(df_split.index)]
    df_split.reset_index(drop=True, inplace=True)
    #TODO: special case for different iterations
    df_split['iter_id'] = 0
    print(f'[INFO] Post Duplicate Length -> {len(df_split)}')
    
    # COMPLEXITY
    complexity, m_complexity, df_split = cp.run(df_split, estimate_clusters)

    ## Merge with df_all
    # col_names = df_all.columns.values + df_split.columns.values
    df_all = pd.concat([df_all, df_split], sort=False, axis=1) #, ignore_index=True) #, names=col_names)

    if m_complexity is None:
        ##TODO: Run Clustering for stratified labeling split
        pass

    # SPLIT
    df_splits = sp.apply_split(df_split, _fn, complexity, labelers)
    
    return df_all, df_splits

