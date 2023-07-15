# This repository contains the code for time-series (TS) classification with various state-of-the-art TS classification models. 

Entire pipeline is developed in a way such that an easy integration of ***mlflow***, ***hydra*** and ***optuna sweeper*** is possible.

1. The simplest way to run a model on a specific dataset is to run the following command in the terminal, in the root directory of the repository:
```bash
python main.py
```
This will run a model on a dataset specified in the config file `main_config.yaml`, located in the `config` directory. 

2. To optimize hyperparameters of a model, run the following command in the terminal, in the root directory of the repository:
```bash
python main.py --multirun
```
This will run a model on a dataset specified in the config file `main_config.yaml`, located in the `config` directory. However, this time, a search space, specified in `config/search_space/model_name` will be used by optuna to find the optimal hyperparameters. A total number of trial is specified in the `main_config.yaml` file.

3. To run a model on a specific dataset, run the following command in the terminal, in the root directory of the repository:
```bash
python main.py "dataset_name=[Handwriting]" 
```
For a multirun case:
```bash
python main.py --multirun "dataset_name=[Handwriting]"  
```
Similarly, other parameters can also be specified in the terminal, and passed as arguments. 

## Requirements


## Authors



