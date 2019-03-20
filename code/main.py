"""
Main function to run UpLabel.


"""
import pandas as pd

# Custom functions
import utils as ut
import complexity as cp

def load_input(data, cols, extras, target='label', 
                    language='de', task='cat', labelers=1,
                    quality=1):
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

    """
    if isinstance(data, str):
        data = pd.read_csv(data, sep='\t', encoding='utf-8')
    
    #check if data is list of DFs or single
    if isinstance(data, (list)):
        print('TODO: LIST OF DATAFRAMES')
        #TODO: merge DFs


    # PREPARE
    df_data = data[cols].copy()
    df_extra = data[extras].copy()
    print(f'[INFO] Input Length -> {len(df_data)}')
    print(f'[INFO] Label Counts: \n{df_data.label.value_counts()}')
    ## Standardize
    df_data.columns = ['text','label','tag']
    ## Drop Duplicates
    df_data.sort_values(by=['label'], inplace=True)
    df_data.drop_duplicates(subset=['text'], inplace=True)
    df_extra = df_extra[df_extra.index.isin(df_data.index)]
    df_data.reset_index(drop=True, inplace=True)
    print(f'[INFO] Post Duplicate Length -> {len(df_data)}')
    
    # COMPLEXITY
    complexity = cp.run(df_data)
    if complexity is None:
        ##TODO: Run Clustering for stratified labeling split
        pass

    # SPLIT
    #TODO:

    return df_data, df_extra

