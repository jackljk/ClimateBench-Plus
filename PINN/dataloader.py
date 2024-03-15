import os
import numpy as np
import pandas as pd
import xarray as xr
from helper_funcs import *

def get_processed_data(args):
    simus = ['ssp126',
            'ssp370',
            'ssp585',
            'hist-GHG',
            'hist-aer']
    X_train = []
    Y_train = []

    for i, simu in enumerate(simus):

        input_name = 'inputs_' + simu + '.nc'
        output_name = 'outputs_' + simu + '.nc'

        # Just load hist data in these cases 'hist-GHG' and 'hist-aer'
        if 'hist' in simu:
            # load inputs 
            input_xr = xr.open_dataset(args["datapath"] + input_name)
                
            # load outputs                                                             
            output_xr = xr.open_dataset(args["datapath"] + output_name).mean(dim='member')
            output_xr = output_xr.assign({"pr": output_xr.pr * 86400,
                                        "pr90": output_xr.pr90 * 86400}).rename({'lon':'longitude', 
                                                                                'lat': 'latitude'}).transpose('time','latitude', 'longitude').drop(['quantile'])
        
        # Concatenate with historical data in the case of scenario 'ssp126', 'ssp370' and 'ssp585'
        else:
            # load inputs 
            input_xr = xr.open_mfdataset([args["datapath"] + 'inputs_historical.nc', 
                                        args["datapath"] + input_name]).compute()
                
            # load outputs                                                             
            output_xr = xr.concat([xr.open_dataset(args["datapath"] + 'outputs_historical.nc').mean(dim='member'),
                                xr.open_dataset(args["datapath"] + output_name).mean(dim='member')],
                                dim='time').compute()
            output_xr = output_xr.assign({"pr": output_xr.pr * 86400,
                                        "pr90": output_xr.pr90 * 86400}).rename({'lon':'longitude', 
                                                                                'lat': 'latitude'}).transpose('time','latitude', 'longitude').drop(['quantile'])

        print(input_xr.dims, simu)
        # Append to list 
        X_train.append(input_xr)
        Y_train.append(output_xr)
    def normalize(data, var, meanstd_dict):
        mean = meanstd_dict[var][0]
        std = meanstd_dict[var][1]
        return (data - mean)/std

    len_historical = 165
    # Compute mean/std of each variable for the whole dataset
    meanstd_inputs = {}

    for var in ['CO2', 'CH4', 'SO2', 'BC']:
        # To not take the historical data into account several time we have to slice the scenario datasets
        # and only keep the historical data once (in the first ssp index 0 in the simus list)
        array = np.concatenate([X_train[i][var].data for i in [0, 3, 4]] + 
                            [X_train[i][var].sel(time=slice(len_historical, None)).data for i in range(1, 3)])
        print((array.mean(), array.std()))
        meanstd_inputs[var] = (array.mean(), array.std())

    # Open and reformat test data 
    X_test = xr.open_mfdataset([args["datapath"] + 'inputs_historical.nc',
                                args["datapath"] + 'inputs_ssp245.nc']).compute()

    # Normalize input data 
    for var in ['CO2', 'CH4', 'SO2', 'BC']: 
        var_dims = X_test[var].dims
        X_test = X_test.assign({var: (var_dims, normalize(X_test[var].data, var, meanstd_inputs))}) 
        
    X_test_np = input_for_training(X_test, skip_historical=False, len_historical=len_historical)
    return X_test, X_test_np


