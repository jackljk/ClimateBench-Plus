import os
import datetime as dt  # Python standard library datetime  module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import RandomizedSearchCV
from eofs.xarray import Eof
import utils
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import random

# Create the random grid
param_grid = {
    'learning_rate': np.linspace(0.01, 0.2, 10),
    'n_estimators': [int(x) for x in np.linspace(start=100, stop=300, num=5)],
    'max_depth': [int(x) for x in np.linspace(5, 20, num=4)],
    'min_child_weight': [int(x) for x in np.linspace(1, 10, num=5)],  
    'subsample': np.linspace(0.8, 1.0, 5),
    'colsample_bytree': np.linspace(0.4, 1.0, 5),
    'gamma': np.linspace(0, 0.5, 6),  
}

def get_rmse(truth, pred):
    weights = np.cos(np.deg2rad(truth.lat))
    return np.sqrt(((truth - pred)**2).weighted(weights).mean(['lat', 'lon'])).data

def xgboost(X_train,y_train,test_X,test_y):
    xgb = xgb.XGBRegressor()
    random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid1,
    n_iter=10,  # Number of parameter settings that are sampled
    scoring='neg_mean_squared_error',  # Use negative RMSE as scoring metric
    cv=5,  # Number of cross-validation folds
    verbose=2,  # Print progress information
    n_jobs=-1  # Use all available CPU cores)
    random_search.fit(X_train, y_train)
    best_xgb = random_search.best_estimator_
    m_out = best_xgb.predict(test_X)
    m_out = pred_tas.reshape(86,96,144)
    return rmse(test_y, m_out)
