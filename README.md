# Twitter Text Classification

This project was created as part of the Machine Learning course [ **CS-433** ] at EPFL.
For this, we developed an end-to-end text classification pipeline that contains data preparation, feature engineering, model training, and performance improvements using classical as well as state-of-the-art methods in text classification. 

## CrowdAI submission

- Name: **TODO**
- ID: **TODO**
- Link: **TODO**

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites

The project was created and tested with the following dependencies:

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

## Project Structure

The project is organized as follows:

    .
    ├── data                        # Root directory all the data
        ├── models                  # Contains all models in binary format
        ├── raw                     # Contains training and testing data
        ├── representations         # Contains text representations
    ├── logs                        # Contains logs
    ├── src                         # Root diractory of our codebase
        ├── experiments             # Contains the code for classical ML experiments
        ├── models                  # Contains classes for ML models
            ├── callbacks           # Contains callback functionalities
        ├── representations         # Contains classes for working with representations
        ├── tweets                  # Contains classes for working with Twitter data
            ├── utils               # Contains utility methods for Twitter data
    ├── tmp                         # Contains temporary files
    ├── utils                       # Contains utility code provided by the ML team
    ├── generate_datasets.py        # Script for creating datasets
    ├── generate_representations.py # Script for creating representations
    ├── README.md                   # Readme file
    ├── requirements.txt            # Requirements file
    ├── run_ml                      # Script for running classical ML experiments
    ├── run.py                      # Script for running the final model
    ├── setup.sh                    # Script for setup

### Data and Models

All of our data can be downloaded from the following link:

Note: We strongly advise to use our downloaded datasets, since it takes too much time for the whole process to complete.

Link: **TODO**

Extract it in the root folder as is.

If you still want to generate your own embeddings, execute:

```
python3 generate_representations.py
```

first, then execute:

```
python3 generate_datasets.py
```

to create the training and test datasets.

## Reproducability

Note: You first have to have a valid directory structure with the necessary data.

If you want to run the final model, execute:

```
python3 run.py
```

**TODO**

### Classical ML

If you want to run the experiments, execute:

```
python3 run_ml.py
```

## Authors

* Mladen Korunoski      mladen.korunoski@epfl.ch
* Ljupche Milosheski    ljupche.milosheski@epfl.ch
* Konstantinos Ragios   konstantinos.ragios@epfl.ch
