# Use miniconda as base image
FROM continuumio/miniconda3

# Install jupyternotebook
RUN conda install -y jupyter

# Create new environment from requirements.yaml
COPY requirements.yaml .
RUN conda env create -f requirements.yaml

# Clone the repo "https://github.com/mijanr/ts_classification_mlflow_hydra_optuna" into the docker image
# First create a directory projects at the home directory
RUN mkdir /home/projects
WORKDIR /home/projects

# Clone the repo
RUN git clone https://github.com/mijanr/ts_classification_mlflow_hydra_optuna.git

# Activate the environment
RUN echo "source activate ts_classification_mlflow_hydra_optuna" > ~/.bashrc
ENV PATH /opt/conda/envs/ts_classification_mlflow_hydra_optuna/bin:$PATH

# Set the working directory
WORKDIR /home/projects/ts_classification_mlflow_hydra_optuna

# Expose the port
EXPOSE 8888
