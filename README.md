# AIMI
Repo for the University of Amsterdam course "Artificial intelligence for medical imaging"

## Contributors
Manin van Ingen
Tina Wünn
Yassine Mashoub
Daniel van de Pavert
Maurice Kingma

## Contents
The python files containing the model architectures and training and validation logic is placed in the `./code` directory. Configs that are used for the different model runs can be find in the `./configs` directory. The jobs used to run the models on the lisa cluster are placed in the `./jobs` directory. Data exploration, testing the Deep Image Prior model and the result analysis are performed using the Jupyter notebooks which can be found in the `./notebooks` directory. Outputs that are used in the reports are placed in te `./output` directory.

## Installing the environment
The conda environment can be installed using `environment.yaml`. Or when using LISA (cluster computer) with the command:
`sbatch ./jobs/install_env.job`

## Running the models
# Deep image prior
The deep image prior model can be runned using the following command:
`python ./code/dip.py ./configs/dip.yaml`

Or using LISA (cluster computer) by running the job via `sbatch`:
`sbatch ./jobs/dip.job`

# Autoencoder
The autoencoder model can be runned using the following command:
`python ./code/ae.py ./configs/data_config.yaml ./configs/model_config.yaml`

Or using LISA (cluster computer) by running the job via `sbatch`:
`sbatch ./jobs/ae.job`