''' The main script for executing classical ML experiments.

Uncomment the lines that you want to execute.
1. It is necessary to have representations on disk
2. It is necessary to have train and test data on disk
3. If 1. and 2. are satisfied, just execute the experiments
'''
import numpy as np
np.set_printoptions(suppress=True)

# ------------------------------------------------------------------
# EXPERIMENTS EXECUTION
# ------------------------------------------------------------------
from src.experiments.executor import ExperimentExecutor
from src.models.classic import *


if __name__ == '__main__':
    ExperimentExecutor.execute(LogisticRegression)
    ExperimentExecutor.execute(SVM)
    ExperimentExecutor.execute(RandomForest)
    ExperimentExecutor.execute(NaiveBayes)
