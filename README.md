# ClimateBench-Plus


## DKL Gaussian Process Regression
To run the code, after creating a basic conda environment, go to the `requirement.txt` file in `DKL Gaussian Process` folder and 
run the following command:
```bash
pip install -r requirements.txt
```

**NOT IMPLEMENTED YET** (Code is currently only in the notebooks as model is still incomplete)
Then, run the following command to run the code:
```bash
python main.py
```
It will run the code and save the results in the `results` folder. 

## XGBoost Regression
To run the code, after creating a basic conda environment, go to the `requirement.txt` file in `XGBoost` folder and 
run the following command:
```bash
pip install -r requirements.txt
```

The preprocessed training and test data can be obtained from Zenodo: (https://zenodo.org/records/7064308). You need the `utils.py` file in `XGBoost` folder to prepare the preprocessed data and save the training and test data by a certain output path.

**NOT IMPLEMENTED YET** (Code is currently only in the notebooks as model is still incomplete)
Then, run the following command to run the code:
```bash
python main.py
```

## CNN LSTM with Physics-Informed Loss
