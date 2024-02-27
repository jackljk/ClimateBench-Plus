from ray import tune

X_train = None
y_train = None
X_test = None
truth = None
train_mean = None
train_std = None



def set_data(x_train, Y_train, x_test, truth_data, trainMean, trainStd):
    global X_train, y_train, X_test, truth, train_mean, train_std
    X_train = x_train
    y_train = Y_train
    X_test = x_test
    truth = truth_data
    train_mean = trainMean
    train_std = trainStd



SEARCH_SPACE = {
    "activation": tune.choice(["relu", "tanh", 'sigmoid']), "bnorm": tune.choice([True, False]), "dropout": tune.choice([True, False]), 'dropout_prob': tune.choice([0.5]),
    "kernel_types": tune.choice([4*['Matern32'], 4*['Matern12'], 4*['Matern52'], 4*['SquaredExponential']]), "active_dim_multiplier": tune.choice([1, 2]), "dim_max": tune.choice([128, 256, 64]), 
    "output_dim": tune.choice([12, 24, 36, 48, 60]), "optimizer_lr": tune.choice([0.01, 0.001])
}     