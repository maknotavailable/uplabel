"""
Log labeling iterations and metrics

Local or from blob storage.
"""
import json
from pathlib import Path

class Log():
    def __init__(self, fn):
        self.fn = fn
        self.logs = []

    def read_log(self):
        """load or create logs"""
        if Path(self.fn).is_file():
            with open(self.fn, 'r') as fn:
                self.logs = json.load(fn)
        else:
            # Initialize log
            self.logs =  dict(iterations=[dict()])

    def write_log(self, name, value, save=True):
        """write logs"""
        self.logs['iterations'][self.iter][name] = value
        if save:
            with open(self.fn, 'w') as fn:
                json.dump(self.logs,fn)

    def set_iter(self, iter):
        print(f'[INFO] Iteration # {iter}')
        self.iter = iter
        
        if len(self.logs['iterations']) != self.iter +1:
            self.logs['iterations'].append(dict(iteration = iter))
        self.write_log('iteration',iter)

