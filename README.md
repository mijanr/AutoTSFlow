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
To run for a specific model:
```bash
python main.py --multirun "dataset_name=[Handwriting]" "models=LSTM_FCN"
```
Model name can be anything that is available in the `codes/models` directory, given corresponding configs are also available.

Similarly, other parameters can also be specified in the terminal, and passed as arguments. 
## Mlruns
All the runs are stored in the `mlruns` directory. To visualize the runs, run the following command in the terminal, in the root directory of the repository:
```bash
mlflow ui
```
This will start a server, and the runs can be visualized in the browser at `localhost:5000`.

## Requirements
requirements.yml file contains all the dependencies required to run the code. To install all the dependencies, run the following command in the terminal, given that anaconda is installed:
```bash
conda env create -f requirements.yaml
```
This will create a conda environment named `ts_cl` with all the dependencies installed.
It insall Pytorch with CPU support. To install Pytorch with GPU support, follow the instructions given [here](https://pytorch.org/get-started/locally/).

## Datasets
This repository uses the datasets from the [UEA & UCR Time Series Classification Repository](https://www.timeseriesclassification.com/). The datasets are automatically downloaded and stored in the `data` directory.

## Models
We use the classification models available in [tsai library](https://timeseriesai.github.io/tsai/). Models can be added to this repository by adding the corresponding config file in the `config` directory, and the corresponding model file in the `codes/models` directory.

## Results
You can find the results in the following table:

<!--START_SECTION:best_runs-->
<!--START_SECTION:best_runs-->

<!--END_SECTION:best_runs-->


<!--END_SECTION:best_runs-->


## Authors
* [**Md Mijanur Rahman**](https://github.com/mijanr)
