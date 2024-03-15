# ClimateBench-Plus

## Website
To see what's our project about, here is the link to website: (https://jackljk.github.io/DSC180B-website/)
## Data
The preprocessed training and test data can be obtained from Zenodo: (https://zenodo.org/records/7064308). 

## DKL Gaussian Process Regression

### Setting up the environment for DKL Gaussian Process
To run the code, first you need to have anaconda install on your machine. then navigate to the `DKL Gaussian Process` folder (or the directory that contains the `environment.yml` file) and run the following command:
```bash
conda env create -f environment.yml
```
This will create a conda environment in the name `CB2` with all the required packages. Then, activate the environment by running the following command:
```bash
conda activate CB2
```
### Running the DKL Gaussian Process
After activating the environment, in the terminal `cd` to the `DKL-Run-Model-Script` folder. In the config file `config.yaml` you can set some of the parameters of the model and whether you want to train a model(`train`) with a set of parameters or perform a hyperparameter search (`search`). More information about the parameters and the settings to run the model can be found in the `config.yaml` file. Then to run the model, run the following command:
```bash
python main.py --config configs/config.yaml 
```
If set to `search` the model will setup a hyperparameter search using the `raytune` and return a csv file for all the trials performed in the directory specified in the `config.yaml` file (default is `hyperparam_results/`). It will also save the predictions of the best model in the directory specified in the `config.yaml` file (default is `model_results/`). If *plot* is set to `True` in the `config.yaml` file, the model will also plot the predictions of the best model and save them in the directory specified in the `config.yaml` file (default is `model_results/`).

### Extras
- Some of my results and predictions from models that I have trained can be found in the `results` directory including the final results that I used for the validation of the model.
- The `notebooks` directory contains the notebooks for each of the Final individual models which I trained and performed my hyperparameter search on.
- The `tests` directory contains my other attempts to implement the Deep Kernel Learning model and which did not work out as expected.

### XGBoost Regression
To run the code, after creating a basic conda environment, go to the `requirement.txt` file in `XGBoost` folder and 
run the following command:
```bash
pip install -r requirements.txt
```
You need the `utils.py` file in `XGBoost` folder to prepare the preprocessed data and save the training and test data by a certain output path.


### Running XGBoost
Replace the parameters, and run the following command to run the model:
```bash
python xgboost_main.py
```

## CNN LSTM with Physics-Informed Loss
To run the code, after creating a basic conda environment, go to the `requirement.txt` file in `DKL Gaussian Process` folder and 
run the following command:
```bash
pip install -r requirements.txt
```

## Running PINN** 
Then, run the `PINN.ipynb' in 'PINN' folder
