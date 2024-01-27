import xarray as xr
import numpy as np


slider = 10 # Temporal window size by year

def getData(filepath, simulations):
    X_train, Y_train = {}, {}

    for i, simu in enumerate(simulations): # Loop over simulations
        # Define input and output file names for each simulation
        input_name = 'inputs_' + simu + '.nc'
        output_name = 'outputs_' + simu + '.nc'

        # Just load hist data in these cases 'hist-GHG' and 'hist-aer'
        if 'hist' in simu:
            # load inputs 
            input_xr = xr.open_dataset(filepath + input_name)
                
            # load outputs                                                             
            output_xr = xr.open_dataset(filepath + output_name).mean(dim='member')
            # Convert the precip values to mm/day
            output_xr = output_xr.assign({"pr": output_xr.pr * 86400,"pr90": output_xr.pr90 * 86400})\
                .rename({'lon':'longitude','lat': 'latitude'}).transpose('time','latitude', 'longitude').drop(['quantile'])
        
        # Concatenate with historical data in the case of scenario 'ssp126', 'ssp370' and 'ssp585'
        else:
            # load inputs 
            input_xr = xr.open_mfdataset([filepath + 'inputs_historical.nc', 
                                        filepath + input_name]).compute()
                
            # load outputs                                                             
            output_xr = xr.concat([xr.open_dataset(filepath + 'outputs_historical.nc').mean(dim='member'),
                                xr.open_dataset(filepath + output_name).mean(dim='member')],dim='time').compute()
            
            # Convert the precip values to mm/day
            output_xr = output_xr.assign({"pr": output_xr.pr * 86400,"pr90": output_xr.pr90 * 86400})\
                .rename({'lon':'longitude', 'lat': 'latitude'}).transpose('time','latitude', 'longitude').drop(['quantile'])

        print(input_xr.dims, simu)

        # Append to dictionary
        X_train[simu] = input_xr
        Y_train[simu] = output_xr
        
    return X_train, Y_train
        
        
def normalize(data, var, meanstd_dict):
    mean = meanstd_dict[var][0]
    std = meanstd_dict[var][1]
    return (data - mean)/std

def unnormalize(data, var, meanstd_dict):
    mean = meanstd_dict[var][0]
    std = meanstd_dict[var][1]
    return data * std + mean


def input_for_training(X_train_xr, skip_historical=False, len_historical=None): 
    
    X_train_np =  X_train_xr.to_array().transpose('time', 'latitude', 'longitude', 'variable').data

    time_length = X_train_np.shape[0]
    # If we skip historical data, the first sequence created has as last element the first scenario data point
    if skip_historical:
        X_train_to_return = np.array([X_train_np[i:i+slider] for i in range(len_historical-slider+1, time_length-slider+1)])
    # Else we just go through the whole dataset historical + scenario (does not matter in the case of 'hist-GHG' and 'hist_aer')
    else:
        X_train_to_return = np.array([X_train_np[i:i+slider] for i in range(0, time_length-slider+1)])
    
    return X_train_to_return


def output_for_training(Y_train_xr, var, skip_historical=False, len_historical=None): 
    Y_train_np = Y_train_xr[var].data
    
    time_length = Y_train_np.shape[0]
    
    # If we skip historical data, the first sequence created has as target element the first scenario data point
    if skip_historical:
        Y_train_to_return = np.array([[Y_train_np[i+slider-1]] for i in range(len_historical-slider+1, time_length-slider+1)])
    # Else we just go through the whole dataset historical + scenario (does not matter in the case of 'hist-GHG' and 'hist_aer')
    else:
        Y_train_to_return = np.array([[Y_train_np[i+slider-1]] for i in range(0, time_length-slider+1)])
    
    return Y_train_to_return