#from hyperopt import fmin, tpe, Trials, STATUS_OK, hp

from hyperopt import fmin, tpe, Trials, STATUS_OK, hp
from hyperopt.pyll.base import scope
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
import multiprocessing
import concurrent.futures
import warnings

# Set PyTorch device to CPU
device = torch.device('cpu')

warnings.filterwarnings("ignore")



# Import the function from data_preprocessing.py
from data_preprocessing import process_and_split_data


# Function to create the TabNet model
def create_model(params):
    """
    Function to create and return a TabNetRegressor model with specified parameters.
    """
    tabnet_model = TabNetRegressor(
        optimizer_fn=torch.optim.Adam,  # Using Adam optimizer
        optimizer_params=dict(lr=params['learning_rate']),  # Learning rate
        n_d=params.get('n_d', 8),  # Dimension of decision prediction layer
        n_a=params.get('n_a', 8),  # Dimension of attention layer
        n_steps=params.get('n_steps', 3),  # Number of decision steps
        gamma=params.get('gamma', 1.5),  # Relaxation factor
        lambda_sparse=params.get('lambda_sparse', 1e-3)  # Feature sparsity regularization
    )
    
    return tabnet_model

# Define the objective function for HyperOpt
def objective(params, X_train, y_train, X_test, y_test):
    model = create_model(params)
        
    # Fit the model using TabNet's built-in method
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=int(params['epochs']),
        patience=20,
        batch_size=int(params['batch_size']),
        virtual_batch_size=32,
        num_workers=0,
        drop_last=False
    )
    
    # Get validation loss (mean squared error)
    y_pred = model.predict(X_test)

    # Calculate MSE and R²
    val_loss = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Return both loss and r² for reference
    return {'loss': val_loss, 'r2': r2, 'status': STATUS_OK}

# Hyperparameter space
space = {
    'n_d': scope.int(hp.quniform('n_d', 16, 48, 16)),
    'n_a': scope.int(hp.quniform('n_a', 16, 48, 16)),
    'n_steps': scope.int(hp.quniform('n_steps', 3, 7, 1)),
    'gamma': hp.uniform('gamma', 1.0, 1.5),
    'lambda_sparse': hp.loguniform('lambda_sparse', np.log(1e-4), np.log(1e-2)),
    'learning_rate': hp.loguniform('learning_rate', np.log(5e-4), np.log(5e-3)),
    'batch_size': scope.int(hp.quniform('batch_size', 32, 128, 32)),
    'epochs': scope.int(hp.quniform('epochs', 50, 150, 50))
}

# Hyperparameter space - test
# space = {
#     'n_d': hp.choice('n_d', [16, 32]),  # 2 possible values
#     'n_a': hp.choice('n_a', [16, 32]),  # 2 possible values
#     'n_steps': hp.choice('n_steps', [3]),  # 2 possible values
#     'gamma': hp.choice('gamma', [1.0]),  # 2 possible values
#     'lambda_sparse': hp.choice('lambda_sparse', [1e-5]),  # 2 possible values
#     'learning_rate': hp.choice('learning_rate', [1e-2, 1e-4]),  # 2 possible values
#     'batch_size': hp.choice('batch_size', [64]),  # 2 possible values
#     'epochs': hp.choice('epochs', [10, 100])  # 2 possible values
# }

# Split the data only once
def load_and_split_data():
    df = pd.read_csv('../input_data/NbCrVWZr_data_stoic_creep_equil_filtered_v2_IQR_filtered.csv')
    input_columns = ['Nb', 'Cr', 'V', 'W', 'Zr']
    output_columns = ['Kou Criteria']
    X_train, X_test, y_train, y_test = process_and_split_data(
        df, 
        input_columns, 
        output_columns, 
        threshold=1e-9, 
        apply_sc=True, scaling_method='minmax', 
        apply_qt=True, qt_method='uniform', 
        apply_log1p=False, 
        apply_sqrt=False, 
        test_size=0.10, 
        random_state=42
    )
    return X_train, X_test, y_train, y_test

# Define the function to execute a single optimization run
def run_single_trial(X_train, y_train, X_test, y_test):
    
    trials = Trials()
    
    best = fmin(
        fn=lambda params: objective(params, X_train, y_train, X_test, y_test),
        space=space,
        algo=tpe.suggest,
        max_evals=8,  # Number of evaluations per core
        trials=trials,
        rstate=np.random.default_rng(42),
        # Use a queue size equivalent to the number of cores for parallelism
        verbose=False
    )
    
    return best, trials

# Main execution with multiprocessing
# Main execution with multiprocessing
if __name__ == "__main__":
    # Load and split the data (only once, outside of parallelization)
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Parallel execution of HyperOpt across 4 cores
    num_cores = 8

    from joblib import Parallel, delayed
    
    # Use joblib's Parallel and delayed
    results, trials = Parallel(n_jobs=num_cores)(delayed(run_single_trial)(X_train, y_train, X_test, y_test) for _ in range(num_cores))

    # Collect the best results
    best_results = [result[0] for result in results]
    print("Best hyperparameters across 4 cores:", best_results)

    # Combine all trials from all cores
    #combined_trials = [result[1] for result in results]
    all_trials = Trials()
    #for trial_set in combined_trials:
    #    all_trials.trials.extend(trial_set.trials)
        
    # %%

    # Count the total number of calculations (trials)
    num_calculations = len(all_trials.trials)
    print(f"Total number of calculations (trials): {num_calculations}")
    
    # %%

    # Save the combined trials to a CSV file
    def save_trials_to_csv(trials, filename='hyperopt_trials.csv'):
        trials_data = []
        for trial in trials.trials:
            trial_info = {
                'iteration': trial['tid'],  # Trial ID (iteration number)
                'loss': trial['result']['loss'],  # Loss value
                'r2': trial['result']['r2'],  # Loss value
                'status': trial['result']['status'],  # STATUS_OK or other
            }
            for key, value in trial['misc']['vals'].items():
                trial_info[key] = value[0] if isinstance(value, list) and len(value) > 0 else value
            trials_data.append(trial_info)
        
        trials_df = pd.DataFrame(trials_data)
        trials_df.to_csv(filename, index=False)
        print(f"Trials saved to {filename}")

    # Save the combined trials data
    save_trials_to_csv(all_trials)
    
    # %%
    
    # Load the CSV file for plotting
    trials_df = pd.read_csv('hyperopt_trials.csv')
    
    # Plotting the loss values over the iterations (number of trials)
    plt.figure(figsize=(10, 6))
    plt.plot(trials_df.index, trials_df['loss'], label='Loss over Trials', color='blue')
    
    # Add standard deviation bounds (assuming some uncertainty measure)
    plt.fill_between(trials_df.index, trials_df['loss'] - trials_df['loss'].std(), 
                     trials_df['loss'] + trials_df['loss'].std(), color='blue', alpha=0.2)
    
    # Customize the labels and title
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    plt.title('Hyper-parameter Optimization Process')
    plt.legend()
    
    # Save the plot to a file
    plt.savefig('hyperopt_loss_plot.png')
    
    # Show the plot (optional)
    plt.show()
    
    # Plotting the loss values over the iterations (number of trials)
    plt.figure(figsize=(10, 6))
    plt.plot(trials_df.index, trials_df['r2'], label='Loss over Trials', color='blue')
    
    # Add standard deviation bounds (assuming some uncertainty measure)
    plt.fill_between(trials_df.index, trials_df['r2'] - trials_df['r2'].std(), 
                     trials_df['r2'] + trials_df['r2'].std(), color='blue', alpha=0.2)
    
    # Customize the labels and title
    plt.xlabel('Number of Iterations')
    plt.ylabel('R$^2$')
    plt.title('Hyper-parameter Optimization Process')
    plt.legend()
    
    # Save the plot to a file
    plt.savefig('hyperopt_loss_plot.png')
    
    # Show the plot (optional)
    plt.show()