import os
import json

from tensorflow.keras.callbacks import Callback


class EpochEvaluator(Callback):
    ''' This class represents extension of Keras' Callback class.
    It is used to log the performace of the model after each iteration.
    '''

    def __init__(self, path, retrain = False):
        ''' Constructor of the logger.
        
        Parameters
        ----------
        path: String
            The path where the outputs of the logger should be written.
        
        retrain: boolean
            If True, it considers as if we are re-training the model, and
            thus does not overwrite previous logges performaces.
        '''
        
        # Initialize the path of the model.
        self.path = path
        
        # If we are not re-training the model, remove any previous files.
        if not retrain and os.path.isfile(self.path):
            os.remove(self.path)
        
        return
    
    
    def on_epoch_end(self, epoch, logs = None):
        ''' A function that should be implmented in order the logger to work.
        
        Parameters
        ----------
        epoch: _
            A parameter that we do not need.
        
        logs: dictionary
            A dictionary containing current performance measures of the model.
        '''
        
        # If there are no measures, create a new variable.
        if logs is None:
            logs = {}
        
        # If there were no previous measures, create a new file.
        if not os.path.isfile(self.path):
            with open(self.path, 'w') as fw:
                json.dump({}, fw)
        
        # Load any previous measures from the file.
        with open(self.path, 'r') as fr:
            data = json.load(fr)
        
        # Add the current measures to the list of previous measures.
        for key in logs.keys():
            if key not in data.keys():
                data[key] = []
            data[key].append(str(logs[key]))
        
        # Write the newly updated measures to the file.
        with open(self.path, 'w') as fw:
            json.dump(data, fw)
        
        return