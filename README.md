# Tweet Sentiment Classification

This repository contains code for [EPFL ML Text Classification Challenge 2019](https://www.aicrowd.com/challenges/epfl-ml-text-classification-2019). It is organized as part of [Machine Learning](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/) course.

We developed an end-to-end text classification pipeline that contains data preparation, feature engineering, model training, and performance improvements using classical machine learning as well as deep learning architectures using recent successful approaches in NLP for text classification.


## Final AIcrowd submission

 - User: **ljupche98**
 - Submission ID: **29808**
 - Classification accuracy: **88.3%**
 - F1 score: **88.5%**


## Prerequisites

The project was developed and tested on Ubuntu 19.10, with the following dependencies:

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

Just run `setup.sh` for creating an environment and installing necessary libraries. Afterwards, just execute `source env/bin/activate` to activate the environment.


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
        ├── reproduce               # Contains architectures and weights of the top 5 DL models
        ├── reproduce_train         # Contains architectures, configurations and training scripts of best models
    ├── tmp                         # Folder for temporary data
    ├── src                         # Root diractory of our codebase
        ├── experiments             # Contains the code for classical ML experiments
        ├── models                  # Contains classes for ML models
            ├── callbacks           # Contains callback functionalities
        ├── representations         # Contains classes for working with representations
        ├── tweets                  # Contains classes for working with Twitter data
            ├── utils               # Contains utility methods for Twitter data
    ├── utils                       # Contains utility code provided by the ML team
    ├── config.json                 # The current configuration for the DL model. Used by train_deep_model.py.
    ├── generate_datasets.py        # Script for creating datasets
    ├── generate_representations.py # Script for creating representations
    ├── README.md                   # Readme file
    ├── requirements.txt            # Requirements file
    ├── run_ml.py                   # Script for running classical ML experiments
    ├── run.py                      # Script for running the final model
    ├── setup.sh                    # Script for setup
    ├── train_deep_model.py         # Train any of the supported deep learning models using config.json


## Data

The steps to acquire the appropriate datasets are described bellow. 

### Reproducibility data

The data necessary to reproduce our best submission can be downloaded from the following link: [**>> Click here <<**](https://drive.google.com/open?id=1lJj3EbVhRwUybW6zPg5e9KNXVwH3r1Mr)

The file **must be extracted in the root** of this repository. Check the project structure above for more details.

***Note**: The link is only accessible by EPFL emails.*

### All data

The data necessary to execute the additional experiments can be downloaded from the following link: [**>> Click here <<**](https://drive.google.com/open?id=1ZVDPSf5iijlyq_Kc_Nem8DfBVGkcJACj)

The file **must be extracted in the root** of this repository. Check the project structure above for more details.

These datasets contain the required word representations, train and test datasets, and model parameters (network architectures and weights).

***Note**: The link is only accessible by EPFL emails.*

### Generating data

If you want to generate your own word representations, execute:

```
python3 generate_representations.py
```

Afterwards, you have to generate train and test datasets, by executing:

```
python3 generate_datasets.py
```

Then, you can execute the additional experiments or train the deep learning model on the newly generated datasets.

***Note**: These procedures take long time to complete. We strongly advise to use the provided links.*


## Reproducibility

Make sure you have the right datasets in the appropriate folders in order to reproduce our best result or run the additional experiments.

### Reproducing AIcrowd submission

***Note**: For this you need **reproducibility data**.*

1. Download the data and install the required libraries (see above for details).
2. Make sure the folder structure is the same as described above.
3. Make the current working directory of the terminal equal to the root of this repository.
4. Execute ```python3 run.py```.
5. The final file with the predictions is located at ```logs/reproduce/final_predictions.csv```. It is equal to the final submission on AIcrowd.

### Reproducing training procedure

***Note**: For this you also need **reproducibility data**.*

Since our final prediction is ensemble of 5 neural networks, you need to train all of them. The training procedure is as follows:

* For every folder (model) in ```logs/reproduce_train```, do the following:
  * Copy the architecture file (```model.json```) in ```logs/<name of the model>```.
  * Copy the training script (```train_deep_model.py```) and configuration file (```config.json```) in the root of the repository.
  * Make sure your current working directory is the root of the repository.
  * Execute ```python3 train_deep_model.py```.
  * The newly trained weights are available at ```logs/<name of the model>```. Copy that folder to ```logs/reproduce```.
* Make the current working directory of the terminal equal to the root of this repository.
* Execute ```python3 run.py```.

### Additional experiments

***Note**: For this you need **all data**.*

If you want to run the classical machine learning models, execute:

```
python3 run_ml.py
```

If you are interested in only executing a subset of the experiments, comment the appropriate lines in this script.

***Note**: These experiments take long time to complete.*

## Training a new model

***Note**: For this you need **reproducibility data**.*

If you want to train a new deep learning model and get its predictions, update ```config.json``` with the desired parameters and the variable params in ```train_deep_model.py```, then execute:
```
python3 train_deep_model.py
```

## Authors

* Mladen Korunoski      mladen.korunoski@epfl.ch
* Ljupche Milosheski    ljupche.milosheski@epfl.ch
* Konstantinos Ragios   konstantinos.ragios@epfl.ch
