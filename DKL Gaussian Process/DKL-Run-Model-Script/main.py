from arguments import arguments
from model import train_model, run_tuner
from CONSTANTS import *
import json
from graphs import *
import ray
import multiprocessing as mp
from dataloader import get_processed_data
import os
from helper_funcs import TNRMSE



def main(args):
    X_train, y_train, X_test, truth, train_mean, train_std = get_processed_data(args)
    print("*"*50)
    print("Done Pre-processing data")
    print("*"*50)

    data = {
        "X_train": X_train, "X_test": X_test, 'y_train': y_train, "truth": truth, "train_mean": train_mean, "train_std": train_std
    }
    
    
    if args['run_type'] == 'search':

        print("*"*50)
        print("Begining Hyperparameter Search")
        print("*"*50)
        
        
        # Run the search
        results = run_tuner(args, SEARCH_SPACE, data, args['no_search_runs'])
        print("*"*50)
        print("Done Running the search")
        print("*"*50)

        # Get results
        df_configs = results.get_dataframe()
        df_config_sorted = df_configs.sort_values(by='nrmse')
        df_config_sorted_configs = df_config_sorted[[col for col in df_config_sorted.columns if 'config' in col]]
        # Best result
        best_config = results.get_best_result()
        # print the best config 
        if verbose:
            print("*"*50)
            print(f"Best Config: Saved at {args['hyperparam_output_dir'] + args['variable'] + '_best_config.json'}")
            print(f"Best Config: {best_config.config}")
            print("*"*50)
            print("\n\n")
            print("*"*50)
            print("Hyperparameter Search Results: Saved at {args['hyperparam_output_dir'] + args['variable'] + '_search_results.csv'}")
            print("*"*50)


        # Save the best config
        with open(args['hyperparam_output_dir'] + args['variable'] + '_best_config.json', 'w') as f:
            json.dump(best_config, f)
        print("*"*50)
        print("Done Saving the best config")
        # Save the results
        df_config_sorted.to_csv(args['hyperparam_output_dir'] + args['variable'] + '_search_results.csv')
        print("Saved the search results")
        print("*"*50)

    elif args['run_type'] == 'train':
        config = json.load(open(args['model_hyperparameters'], 'r'))
        # Train the model
        mean, std = train_model(config, return_pred=True)
        if verbose:
            print("*"*50)
            print("Done Training the model")
            print("*"*50)
        # Save the model
        mean.to_netcdf(args['model_output_dir'] + args['variable'] + '-mean.nc')
        std.to_netcdf(args['model_output_dir'] + args['variable'] + '-std.nc')
        if verbose:
            print("*"*50)
            print("Saved the model")
            print("*"*50)

        if args['TNRMSE']:
            print("*"*50)
            tnrse = TNRMSE(truth, mean)
            print(f"TNRMSE (2080-2100): {tnrse}")
            print("*"*50)
    else:
        print("*"*50)
        print("Invalid run type")
        print("*"*50)
        return

    if args['plot']:
        # Plot the results
        plot_maps(mean, truth, args)
        plot_timeseries(mean, truth, args)
        print("*"*50)
        print("Done Plotting the results")
        print("*"*50)

if __name__ == "__main__":
    print("*"*50)
    print("Starting Ray Instance")
    print("*"*50)
    ray.init(num_cpus=1, num_gpus=1)
    print("*"*50)
    print("Ray Instance Initialized")
    print("*"*50)
    
    # Get the arguments
    args, verbose = arguments()
    # Check if the directory exists
    if not os.path.exists(args['hyperparam_output_dir']):
        # Create the directory
        os.makedirs(args['hyperparam_output_dir'])

    if not os.path.exists(args['model_output_dir']):
        # Create the directory
        os.makedirs(args['model_output_dir'])


    if verbose:
        print('Arguments:', args)
        
    main(args)

    print("*"*50)
    print("Done")
    print("*"*50)