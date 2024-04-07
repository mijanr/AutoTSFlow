# AutoTSFlow: An automated code-base for time-series classification
This repository contains the code for time-series (TS) classification with various state-of-the-art TS classification models. 
The entire pipeline is developed for easy integration of ***mlflow***, ***hydra***, and ***optuna sweeper***, facilitating efficient experimentation, hyperparameter tuning, and configuration management.

The pipeline is developed in a modular way, where the models, datasets, and configurations can be easily added or modified.


1. The simplest way to run a model on a specific dataset is to run the following command in the terminal in the root directory of the repository:
```bash
python main.py
```
This will run a model on a dataset specified in the config file `main_config.yaml`, located in the `config` directory. 

2. To optimize the hyperparameters of a model, run the following command in the terminal in the root directory of the repository:
```bash
python main.py --multirun
```
This will run a model on a dataset specified in the config file `main_config.yaml`, located in the `config` directory. However, this time, a search space specified in `config/search_space/model_name` will be used by optuna to find the optimal hyperparameters. A total number of trials is specified in the `main_config.yaml` file.

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

## Docker 
To run the code in a docker container, run the following command in the terminal, in the root directory of the repository:
```bash
docker build -t ts_cl .
docker run -it ts_cl
```
This will build a docker image named `ts_cl`, and run a container with the image. The code can be run in the container as described above.

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
You can find the results in the following table. Each cell contains the accuracy of the corresponding model on the corresponding dataset. The results are obtained by running the models with the optimal hyperparameters found by optuna.

<!--START-->
| Dataset               |   GRU_FCN |   InceptionTime |      LSTM |   LSTM_FCN |
|:----------------------|----------:|----------------:|----------:|-----------:|
| ECG200                |  0.91     |       0.91      | 0.82      |  0.92      |
| HandMovementDirection |  0.459459 |     nan         | 0.472973  |  0.486486  |
| Handwriting           |  0.101176 |       0.0952941 | 0.0541176 |  0.0752941 |
| ItalyPowerDemand      |  0.970845 |       0.969874  | 0.559767  |  0.910593  |
<!--END-->


## Authors
* [**Md Mijanur Rahman**](https://github.com/mijanr)
