import gpflow
from gpflow.mean_functions import Constant
from helper_funcs import make_feature_extractor, basekernel, TNRMSE
import tensorflow as tf
import numpy as np
import xarray as xr
import gpflow
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from CONSTANTS import *



class DeepKernel(gpflow.kernels.Kernel):
    def __init__(self, feature_extractor, base_kernel, input_dim):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.base_kernel = base_kernel
        self.input_dim = input_dim

    def K(self, X, X2=None):
        # Transform X and X2 using the neural network
        X_transformed = self.feature_extractor(X)
        X2_transformed = self.feature_extractor(X2) if X2 is not None else X2
        # Compute the kernel using the transformed inputs
        return self.base_kernel(X_transformed, X2_transformed)

    def K_diag(self, X):
        X_transformed = self.feature_extractor(X)
        return self.base_kernel.K_diag(X_transformed)


def train_model(config, data, return_pred=False):  # ①
    input_dim = data["X_train"].shape[1]  # Number of features in X

    
    output_dim = config["output_dim"]
    # Feature extractor for deep kernel
    feature_extractor = make_feature_extractor(
        config["dim_max"], config["activation"], input_dim, config["output_dim"], config["dropout_prob"], bnorm=config["bnorm"], dropout=config["dropout"]
    )
    
    # Freeze the neural network layers to make them non-trainable in GPflow's optimization process
    for layer in feature_extractor.layers:
        layer.trainable = True
    
    # Define kernel
    base_kernel = basekernel(
        config["kernel_types"], config["active_dim_multiplier"]
    )
    deep_kernel = DeepKernel(feature_extractor=feature_extractor, base_kernel=base_kernel, input_dim=input_dim)

    mean_function = Constant() # Define Mean function

    optimizer  = tf.keras.optimizers.Adam(learning_rate=config["optimizer_lr"])# Define opt

    model = gpflow.models.GPR(data=(data["X_train"].astype(float), data["y_train"].astype(float)), kernel=deep_kernel, mean_function=mean_function)
    

    # custom optimizer
    @tf.function
    def optimization_step():
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(model.trainable_variables)
            loss = -model.log_marginal_likelihood()
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss


    # Training
    tolerance, patience = 1e-6, 10  # Define tolerance and patience

    # Initialize variables for early stopping mechanism
    best_loss, patience_counter, max_iterations = float('inf'), 0, 10000
    
    # Training loop
    for iteration in range(max_iterations):  # Number of optimization steps
        loss = optimization_step()
    
        # Early stopping logic
        if best_loss - loss < tolerance:
            patience_counter += 1
        else:
            patience_counter = 0
            best_loss = loss
    
        if patience_counter >= patience:
            print(f"Stopping training after {iteration + 1} iterations due to convergence.")
            break

    # Eval
    standard_posterior_mean, standard_posterior_var = model.predict_y(data["X_test"].values)
    posterior_mean = standard_posterior_mean * data["train_std"] + data["train_mean"]
    posterior_std = np.sqrt(standard_posterior_var) * data["train_std"]

    # put output back into xarray format for calculating RMSE/plotting
    posterior_pr = np.reshape(posterior_mean, [86, 96, 144])
    posterior_std = np.reshape(posterior_std, [86, 96, 144])
    
    posterior_data = xr.DataArray(posterior_pr, dims=data["truth"].dims, coords=data["truth"].coords)
    posterior_std_data = xr.DataArray(posterior_std, dims=data["truth"].dims, coords=data["truth"].coords)

    total_NRMSE = TNRMSE(data["truth"], posterior_data)

    if return_pred:
        return posterior_data, posterior_std_data


    return {'nrmse':total_NRMSE}


def run_tuner(args, search_space, data, num_samples=50):
    algo = HyperOptSearch()

    tuner = tune.Tuner(  # ③
        tune.with_resources(
            lambda config: train_model(config, data),
            resources={"cpu": 1, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric="nrmse",
            mode="min",
            num_samples=num_samples,
            search_alg=algo,
        ),
        param_space=search_space,
    )

    results = tuner.fit()
    return results


