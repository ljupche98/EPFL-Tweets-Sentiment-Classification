import os
import json

from tensorflow.keras.callbacks import Callback


class EpochEvaluator(Callback):


    def __init__(self, path, retrain = False):
        self.path = path
        
        if not retrain:
            os.remove(self.path)
        
        return
    
    
    def on_epoch_end(self, epoch, logs = None):
        if logs is None:
            logs = {}
        
        if not os.path.isfile(self.path):
            with open(self.path, 'w') as fw:
                json.dump({}, fw)
        
        with open(self.path, 'r') as fr:
            data = json.load(fr)
        
        for key in logs.keys():
            if key not in data.keys():
                data[key] = []
            data[key].append(str(logs[key]))
        
        with open(self.path, 'w') as fw:
            json.dump(data, fw)
        
        return