"""
UpLabel.

--Authors: Martin Kayser, Timm Walz. 


"""
import yaml
import pandas as pd

# Custom functions
import utils as ut
import complexity as cp
import split as sp
import join as jo
import log as lg

class Main():
    def __init__(self, project, debug_iter_id=None):
        #Load logs
        fn_log = '../task/'+project+'/log.json'
        self.log = lg.Log(fn_log)
        self.log.read_log()
        
        #Load config
        with open('../task/'+project+'/params.yaml', 'r') as stream:
            params = yaml.safe_load(stream)
        self.data_dir = params['data']['dir']

        # Run iteration
        if debug_iter_id is not None:
            # Debug mode
            self.log.set_iter(debug_iter_id)
            self.load_input(params['data']['dir']+params['data']['source'],
                            params['data']['cols'],
                            params['data']['extras'],
                            language = params['parameters']['language'],
                            labelers = params['parameters']['labelers'],
                            quality = params['parameters']['quality'],
                            estimate_clusters = params['parameters']['estimate_clusters'])
        elif any('iteration' in it.keys() for it in self.log.logs['iterations']):
            self.log.set_iter(len(self.log.logs['iterations']))
            self.load_input(params['data']['dir']+params['data']['source'],
                            params['data']['cols'],
                            params['data']['extras'],
                            language = params['parameters']['language'],
                            labelers = params['parameters']['labelers'],
                            quality = params['parameters']['quality'],
                            estimate_clusters = params['parameters']['estimate_clusters'])            
        else:
            self.log.set_iter(0)
            self.load_input(params['data']['dir']+params['data']['source'],
                            params['data']['cols'],
                            params['data']['extras'],
                            language = params['parameters']['language'],
                            labelers = params['parameters']['labelers'],
                            quality = params['parameters']['quality'],
                            estimate_clusters = params['parameters']['estimate_clusters'])


    def prepare_data(self, data, cols, extras):
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
        df_split['iter_id'] = self.log.iter
        print(f'[INFO] Post Duplicate Length -> {len(df_split)}')
        return df_all, df_split

    def load_input(self, path, cols, extras, target='label', 
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
        - df_all (dataframe) : entire dataset, includiong all metadata
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
        df_all, df_split = self.prepare_data(data, cols, extras)
        
        ### COMPLEXITY ###
        complexity, m_complexity, df_split = cp.run(df_split, estimate_clusters)

        ## Log results
        self.log.write_log('complexity', complexity)
        self.log.write_log('performance', complexity)
        self.log.write_log('data_length', len(df_all))
        self.log.write_log('train_length', len(df_split[~df_split.label.isna()]))
        # END

        ## Merge with df_all
        df_all = pd.concat([df_all, df_split], sort=False, axis=1)
        df_all.to_csv('.'.join(path.split('.')[:-1]) + f'-it_{self.log.iter}-residual.txt', sep='\t',encoding='utf-8')

        ### SPLIT ###
        df_splits = sp.apply_split(df_split, path, complexity, labelers, iter_id = self.log.iter)
    
        return df_all, df_splits