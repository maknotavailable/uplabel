import pandas as pd
import configparser


config = configparser.ConfigParser()
config.read('../config.ini', encoding='utf-8')
data_dir = config['path']['data']

import sys
sys.path.append('../code')
import main

df = pd.read_csv(data_dir+'raw/raw_v1.txt', sep='\t', encoding='utf-8')
df_all, df_splits = main.load_input(data_dir+'raw/raw_v1.txt', 
              cols=['text_b','label','tag'], 
              extras=['entity', 'entity_type', 'comment', 'text'],
              target='label',
              language='de',
              task='cat',
              labelers=2,
              quality=1,
              estimate_clusters=False #force skip cluster estimation, when not enough examples are available
             )