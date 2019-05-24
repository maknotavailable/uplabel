import pandas as pd
import json
# Custom functions
import log as lg

class Reporting():
    def __init__(self, project_id):
        self.project_id = project_id
        # Load logs
        fn_log = '../task/'+project_id+'/log.json'
        log = lg.Log(fn_log)
        log.read_log()
        self.df_logs = pd.read_json(json.dumps(log.logs['iterations']))

    def get_progress(self):
        return self.df_logs

    def plot(self, col):
        return self.df_logs[col].plot()

    def plot_multiple(self, cols):
        #TODO: compare plots (eg complexity vs performance vs train length)
        pass
            