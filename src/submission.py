import pandas as pd


DATA_DIR = 'data/'


def generate(pred):
    ''' Sumbission generation procedure.

    Parameters
    ----------
    pred: ndarray
        The predictions to store
    '''
    pred[pred == 0] = -1
    df = pd.read_csv(DATA_DIR + 'sample_submission.csv')
    df['Prediction'] = pred
    df['Prediction'] = df['Prediction'].astype(int)
    df.to_csv(DATA_DIR + 'submission.csv', index=False)
