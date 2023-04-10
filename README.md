# Similarity and competence measures to dynamic predictor selection:: a comparative study
 
 ## Project description

This project evaluates the impact of alternative measures on dynamic predictor selection performance to predict one-step-ahead microservices time series.

# Installation  
  
## How to install the project?

    $ virtualenv venv
    $ source venv/bin/activate
    $ pip3 install -r requirements.txt
    $ apt install zip -y; unzip pickle.zip;
    
## Project Files

Summary of the main repository files.


| Files                      | Content description                                                                 |
|----------------------------|----------------------------------------------------------------------------------   |
| [series-descriptions.csv](blob/main/series-descriptions/series-descriptions.csv)| Description of the datasets.   |
| [result](results)                                                               | MSE restults.                  |
| [pickle](pickle)                                                                | Trained models saved in .pickle|

## How to regenerate results using pickle models?

|  File                                                   | File description                                      |
|---------------------------------------------------------|-------------------------------------------------------|
| [competences.py](/blob/main/competences.py)             | Competence measures.                                  |
| [dynamic_selection.py](blob/main/dynamic_selection.py)  | Run the DS dynamic selection algorithm.               |
| [results.py](blob/main/results.py)                      | Generates csv results from models.                    |
| [generate_oracle.py](blob/main/generate_oracle.py)      | Generate results from Oracle.                         |
| [similarities.py](blob/main/similarities.py)            | Similarity measures.                                  |
| [train_models.py](blob/main/train_models.py)            | Train monolithic models and the homogeneous pool.     |
