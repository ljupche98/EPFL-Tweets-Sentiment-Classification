# Tweet Sentiment Classification

This repository contains code for [EPFL ML Text Classification Challenge 2019](https://www.aicrowd.com/challenges/epfl-ml-text-classification-2019).
It is organized as part of [Machine Learning](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/) course.
We developed a pipeline for tweet preprocessing, feature engineering and word representation, classical machine learning and deep learning models using recent successful approaches in NLP.


## Final AIcrowd submission

 - User: **ljupche98**
 - Submission ID: **29808**
 - Classification accuracy: **88.3%**
 - F1-score: **88.5%**


## Reproducibility

1. Download the data and install the required libraries (see below for details).
2. Make sure the folder structre is the same as described below.
3. Make the current working directory of the terminal equal to the root of this repository.
4. Execute ```python3 run.py```.
5. The final file with the predictions is located at ```logs/reproduce/final_predictions.csv```. It is equal to the final submission at AIcrowd.

### Additional results
If you want to run the classical machine learning models, execute:
```
python3 run_ml.py
```

If you want to train a deep learning model and get its predictions, update ```config.json``` with the desired parameters and the variable params in ```train_deep_model.py```, then execute:
```
python3 train_deep_model.py
```


## Prerequisites

The project was developed and tested with the following dependencies:

### Libraries

```
- Cython
- regex
- numpy
- scipy
- pandas
- fasttext
- sent2vec
- sklearn
- tensorflow
- keras
```

Just run `setup.sh` for creating environment and installing necessary libraries. Afterwards, just execute `source env/bin/activate`.

### Data and Models

#### Reproducibility data

The **necessary data to reproduce our best submission** can be downloaded from the following link: [**>> Click here <<**](https://drive.google.com/open?id=1Z7eNpc3GVCLjG_iHSPxM0Gis7oBJERtZ)

The file must be extracted in the root of this repository. Check the project structure below for more details.

#### Additional adata

The whole data (word embeddings) and models (network architectures and weights) can be downloaded from the following link: [**>> Click here <<**](https://drive.google.com/open?id=1CHvwPWkRB5ka3Iox7LSn4KMu5p2_Nlt-)

The file must be extracted in the root of this repository. Check the project structure below for more details.

It is needed to test the performance of all our models mentioned in the report.

#### Additional word representations

If you want to generate additional own embeddings, execute
```
python3 generate_representations.py
python3 generate_datasets.py
```


## Project Structure

The project is organized as follows:

    .
    ├── data                        # Root directory all the data
        ├── models                  # Contains all models in binary format
        ├── raw                     # Contains training and testing data
        ├── representations         # Contains text representations
    ├── figures                     # Cotains the figures used in the report
    ├── logs                        # Contains logs
        ├── fasttext                # Contains architectures and weights for our fasttext DL models
        ├── GRU                     # Contains the architecture and weight of the last GRU model
        ├── LSTM                    # Contains the architecture and weight of the last LSTM model
        ├── LSTM+GRU                # Contains the architecture and weight of the last LSTM + GRU model
        ├── reproduce               # A MUST HAVE for reproducibility. Contains architectures and weights of the top 5 DL models
    ├── src                         # Root diractory of our codebase
        ├── experiments             # Contains the code for classical ML experiments
        ├── models                  # Contains classes for ML models
            ├── callbacks           # Contains callback functionalities
        ├── representations         # Contains classes for working with representations
        ├── tweets                  # Contains classes for working with Twitter data
            ├── utils               # Contains utility methods for Twitter data
    ├── utils                       # Contains utility code provided by the ML team
    ├── README.md                   # Readme file
    ├── config.json                 # The current configuration for the DL model. Used by train_deep_model.py.
    ├── generate_datasets.py        # Script for creating datasets
    ├── generate_representations.py # Script for creating representations
    ├── requirements.txt            # Requirements file
    ├── run_ml.py                   # Script for running classical ML experiments
    ├── run.py                      # Script for running the final model
    ├── setup.sh                    # Script for setup
    ├── train_deep_model.py         # Uses the configuration in config.json to train any of the supported deep learning models


## Authors

* Mladen Korunoski      mladen.korunoski@epfl.ch
* Ljupche Milosheski    ljupche.milosheski@epfl.ch
* Konstantinos Ragios   konstantinos.ragios@epfl.ch
