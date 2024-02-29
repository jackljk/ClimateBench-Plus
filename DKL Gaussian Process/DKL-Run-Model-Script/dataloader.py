import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import colors
from eofs.xarray import Eof
import tensorflow as tf
import gpflow
from esem import gp_model
import seaborn as sns
import cartopy.crs as ccrs
from helper_funcs import *


train_files = ["ssp126", "ssp370", "ssp585", "historical", "hist-GHG"]

def get_processed_data(args):
    datapath = args.datapath
    var = args.variable

    # Create training and testing arrays
    X_train, eof_solvers = create_predictor_data(train_files)
    y_train = create_predictdand_data(train_files)['var'].values.reshape(-1, 96 * 144)

    X_test = get_test_data('ssp245', eof_solvers)
    Y_test = xr.open_dataset(datapath + 'outputs_ssp245.nc').compute()
    truth = Y_test["var"].mean('member')

    # Drop rows including nans
    nan_train_mask = X_train.isna().any(axis=1).values
    X_train = X_train.dropna(axis=0, how='any')
    y_train = y_train[~nan_train_mask]
    assert len(X_train) == len(y_train)

    nan_test_mask = X_test.isna().any(axis=1).values
    X_test = X_test.dropna(axis=0, how='any')
    truth = truth[~nan_test_mask]

    # Standardize predictor fields requiring standardization (non-EOFs)
    train_CO2_mean, train_CO2_std = X_train['CO2'].mean(), X_train['CO2'].std()
    train_CH4_mean, train_CH4_std = X_train['CH4'].mean(), X_train['CH4'].std()

    X_train.CO2 = (X_train.CO2 - train_CO2_mean) / train_CO2_std
    X_train.CH4 = (X_train.CH4 - train_CH4_mean) / train_CH4_std

    X_test.CO2 = (X_test.CO2 - train_CO2_mean) / train_CO2_std
    X_test.CH4 = (X_test.CH4 - train_CH4_mean) / train_CH4_std
    
    # Standardize predictand fields
    train_mean, train_std = y_train.mean(), y_train.std()
    y_train = (y_train - train_mean) / train_std

    if var in ['pr', 'pr90']:
        truth *= 86400


    return X_train, y_train, X_test, truth, train_mean, train_std



