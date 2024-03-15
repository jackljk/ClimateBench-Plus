from arguments import arguments
import xarray as xr
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, Input, Reshape, AveragePooling2D, TimeDistributed, LSTM, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from dataloader import get_processed_data
from helper_funcs import PINN_loss


def main(args):
    X_test, X_test_np = get_processed_data(args)
    print("*"*50)
    print("Done Pre-processing data")
    print("*"*50)

    model = load_model('built_models/PINN_{}'.format(args["variable"]), custom_objects={"PINN_loss": PINN_loss})

    # Make predictions using trained model 
    m_pred = model.predict(X_test_np)

    # Reshape to xarray 
    m_pred = m_pred.reshape(m_pred.shape[0], m_pred.shape[2], m_pred.shape[3])
    m_pred = xr.DataArray(m_pred, dims=['time', 'lat', 'lon'], coords=[X_test.time.data[10-1:], X_test.latitude.data, X_test.longitude.data])
    xr_prediction = m_pred.transpose('lat', 'lon', 'time').sel(time=slice(2015,2101)).to_dataset(name=args["variable"])

    if args["variable"]=="pr90" or args["variable"]=="pr":
        xr_prediction = xr_prediction.assign({args["variable"]: xr_prediction[args["variable"]] / 86400})

    # Save test predictions as .nc 
    if args["variable"] == 'diurnal_temperature_range':
        xr_prediction.to_netcdf("predictions/" + 'outputs_ssp245_predict_dtr.nc', 'w')
    else:
        xr_prediction.to_netcdf("predictions/" + 'outputs_ssp245_predict_{}.nc'.format(args["variable"]), 'w')
    xr_prediction.close()
    print("Done predicting for {}".format(args["variable"]))

if __name__ == "__main__":
    
    # Get the arguments
    args, verbose = arguments()

    if verbose:
        print('Arguments:', args)
main(args)

print("*"*50)
print("Done")
print("*"*50)