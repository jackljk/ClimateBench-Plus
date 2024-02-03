import numpy as np
import pandas as pd
import xarray as xr
from eofs.xarray import Eof
import os
import datetime as dt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import RandomizedSearchCV
import utils
# data_path = './data/train_val/'
data_path = "CONFIGURE_ME"

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 300, num = 5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5,55, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [5, 10, 15, 25]
# Minimum number of samples required at each leaf node
min_samples_leaf = [4, 8, 12]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
def get_rmse(truth, pred):
    weights = np.cos(np.deg2rad(truth.lat))
    return np.sqrt(((truth - pred)**2).weighted(weights).mean(['lat', 'lon'])).data

def random_forest(X_train,y_train,test_X,test_y):
    reg = RandomForestRegressor(random_state=0)
    rf = reg.fit(X_train,y_train)
    rf_random = RandomizedSearchCV(estimator = reg, param_distributions = random_grid, n_iter = 29, cv = 3, verbose=2, n_jobs = -1)
    rf_random.fit(X_train,y_train)
    print(rf_random.best_params_)
    m_out = rf.predict(test_X)
    return rmse(test_y, m_out)
    
    
    
    
