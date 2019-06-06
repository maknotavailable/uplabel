"""
UpLabel.

--Authors: Martin Kayser, Timm Walz. 


"""
import yaml
import pandas as pd
import numpy as np

# Custom functions
import utils as ut
import complexity as cp
import split as sp
import join as jo
import log as lg

class Main():
    def __init__(self, project, debug_iter_id=None):
        print(f'Starting UpLabel >>>>\tProject {project.upper()}')

        # Load logs
        fn_log = '../task/'+project+'/log.json'
        self.log = lg.Log(fn_log)
        self.log.read_log()
        
        # Load config
        with open('../task/'+project+'/params.yaml', 'r') as stream:
            params = yaml.safe_load(stream)
        self.data_dir = params['data']['dir']

        # Determine iteration
        if debug_iter_id is not None:
            # Debug mode
            self.log.set_iter(debug_iter_id)
        elif any('iteration' in it.keys() for it in self.log.logs['iterations']):
            self.log.set_iter(len(self.log.logs['iterations']))       
        else:
            self.log.set_iter(0)
        
        self.max_split = params['parameters']['max_split_size']

        # Run iteration
        self.run(params['data']['dir']+params['data']['source'],
                        params['data']['cols'],
                        params['data']['extras'],
                        language = params['parameters']['language'],
                        labelers = params['parameters']['labelers'],
                        quality = params['parameters']['quality'],
                        estimate_clusters = params['parameters']['estimate_clusters'])

    def prepare_data(self, data, cols, extras):
        df_split = data.copy() #[cols]
        ## Standardize
        if 'tag' not in df_split.columns:
            df_split['tag'] = ''
        if 'comment' not in df_split.columns:
            df_split['comment'] = ''
        
        print(f'[INFO] Input Length -> {len(df_split)}')
        print(f'[INFO] Label Summary: \n{df_split[df_split.label != ""].label.value_counts()}')
        
        ## Drop Missing
        df_split.dropna(subset=['text'], inplace=True)
        df_split = df_split[df_split.text != '']
        ## Drop Duplicates and missing
        df_split.sort_values(by=['label'], inplace=True)
        df_split.drop_duplicates(subset=['text'], keep='first', inplace=True)
        data = data[data.index.isin(df_split.index)]
        df_split.replace(np.nan, '', regex=True, inplace=True)
        df_split.reset_index(drop=True, inplace=True)
        df_split.reset_index(inplace=True) # create row ID
        df_split['iter_id'] = self.log.iter

        print(f'[INFO] Post Duplicate Length -> {len(df_split)}')

        assert len(df_split[df_split.label == '']) != 0, \
            '[ERROR] Congratulations, all your data is already labeled.'
        assert len(df_split[df_split.label != '']) != 0, \
            '[ERROR] None of the data is labeled. Please provide \
            examples for each category first.'

        return data, df_split

    def run(self, path, cols, extras, target='label', 
                        language='de', task='cat', labelers=1,
                        quality=1, estimate_clusters=True):
        """First contact data loading

        INPUT
        - path (string) : path to data
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
        - df_split (dataframe(s)) : data split for labeling
        """

        if self.log.iter > 0:
            load_iter = self.log.iter - 1
            print(f'[INFO] Loading splits from iteration {load_iter}.')
            data = jo.load_iteration(self.data_dir, load_iter)
        else:
            # Load first iteration
            data = pd.read_csv(path, sep='\t', encoding='utf-8')

        ### PREPARE ###
        __, df_split = self.prepare_data(data, cols, extras)
        
        ### COMPLEXITY ###
        complexity, m_complexity, df_split = cp.run(df_split, estimate_clusters, language)

        ## Log results
        self.log.write_log('complexity', complexity)
        self.log.write_log('performance', complexity)
        self.log.write_log('data_length', len(df_split))
        self.log.write_log('labeled_length', len(df_split[df_split.label != '']))
        self.log.write_log('labels', len(df_split[df_split.label != ''].label.drop_duplicates()))
        # END

        ## Merge with df_all
        # df_all = pd.concat([df_all[extras], df_split], sort=False, axis=1)
        df_all = df_split.copy()
        df_all.drop(['pred_id','pred'], axis=1, inplace=True)
        df_all.to_csv('.'.join(path.split('.')[:-1]) + f'-it_{self.log.iter}-residual.txt', sep='\t',encoding='utf-8')

        ### SPLIT ###
        df_splits = sp.apply_split(df_split, path, complexity, labelers, iter_id = self.log.iter, max_split=self.max_split)
    
        return df_splits