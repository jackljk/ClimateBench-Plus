from args import args
from dataloader import get_processed_data
from model import train_model, run_tuner
from CONSTANTS import *
import json
from graphs import *


def main():
    # Get the arguments
    args = args()

    # Pre-process the data
    X_train, y_train, X_test, truth, train_mean, train_std = get_processed_data(args)
    set_data(X_train, y_train, X_test, truth, train_mean, train_std)
    print("*"*50)
    print("Done Pre-processing data")
    print("*"*50)

    if args.run_type == 'search':
        # Run the search
        results = run_tuner(SEARCH_SPACE, args.num_samples)
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
        print(json.dumps(best_config, indent=4))

        # Save the best config
        with open(args.hyperparam_output_dir + args.var + '_best_config.json', 'w') as f:
            json.dump(best_config, f)
        print("*"*50)
        print("Done Saving the best config")
        # Save the results
        df_config_sorted.to_csv(args.hyperparam_output_dir + args.var + '_search_results.csv')
        print("Saved the search results")
    else:
        config = json.load(open(args.model_config, 'r'))
        # Train the model
        mean, std = train_model(config, return_pred=True)
        # Save the model
        mean.to_netcdf(args.model_output_dir + args.var + '-mean.nc')
        std.to_netcdf(args.model_output_dir + args.var + '-std.nc')
        print("*"*50)
        print("Done Training the model")
        print("*"*50)

    if args.plot:
        # Plot the results
        plot_maps(mean, truth, args)
        plot_timeseries(mean, truth, args)
        print("*"*50)
        print("Done Plotting the results")
        print("*"*50)

if __name__ == "__main__":
    main()