import numpy as np
import xarray as xr
import tensorflow as tf


def input_for_training(X_train_xr, skip_historical=False, len_historical=None): 
    
    X_train_np =  X_train_xr.to_array().transpose('time', 'latitude', 'longitude', 'variable').data

    time_length = X_train_np.shape[0]
    # If we skip historical data, the first sequence created has as last element the first scenario data point
    if skip_historical:
        X_train_to_return = np.array([X_train_np[i:i+10] for i in range(len_historical-10+1, time_length-10+1)])
    # Else we just go through the whole dataset historical + scenario (does not matter in the case of 'hist-GHG' and 'hist_aer')
    else:
        X_train_to_return = np.array([X_train_np[i:i+10] for i in range(0, time_length-10+1)])
    
    return X_train_to_return 


def output_for_training(Y_train_xr, var, skip_historical=False, len_historical=None): 
    Y_train_np = Y_train_xr[var].data
    
    time_length = Y_train_np.shape[0]
    
    # If we skip historical data, the first sequence created has as target element the first scenario data point
    if skip_historical:
        Y_train_to_return = np.array([[Y_train_np[i+10-1]] for i in range(len_historical-10+1, time_length-10+1)])
    # Else we just go through the whole dataset historical + scenario (does not matter in the case of 'hist-GHG' and 'hist_aer')
    else:
        Y_train_to_return = np.array([[Y_train_np[i+10-1]] for i in range(0, time_length-10+1)])
    
    return Y_train_to_return

def PINN_loss(self, y_t, y_p):
    error = y_t - y_p
    # MSE between training and prediction
    mse = tf.reduce_mean(tf.square(error)) 
    # FaIR loss
    C0 = 598
    E = self.input_train[:, :, :, :, 0:1]
    C = C0 + E
    f1 = 4.57
    f2 = 0
    f3 = 0.086
    F = np.add(np.multiply(f1, np.log(np.divide(C,C0))), np.multiply(f3, np.subtract(np.sqrt(C), np.sqrt(C0))))
    qj = 0.3
    T = np.multiply(qj, F)
    dj = 10
    fair= tf.math.divide(T - tf.reduce_mean(y_p), dj)
    fair_mse = tf.reduce_mean(tf.square(fair))
    total = mse + fair_mse
    return total