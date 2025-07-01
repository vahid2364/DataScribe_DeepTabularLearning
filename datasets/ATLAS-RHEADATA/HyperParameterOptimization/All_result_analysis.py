#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 00:49:55 2024

@author: attari.v
"""
import matplotlib.pyplot as plt
import optuna
import seaborn as sns
import numpy as np
import pandas as pd
from optuna.storages import RDBStorage

plt.style.use('default')

# %%

# Function to collect objective values and optionally sort them
def collect_objective_values(study, sort_values=True):
    objective_values = []
    
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None:
            objective_values.append(trial.value)
    
    if sort_values:
        # Sort objective values in descending order
        sorted_objective_values = sorted(objective_values, reverse=True)
        # Generate new trial numbers (1, 2, 3, ...)
        new_trial_numbers = list(range(1, len(sorted_objective_values) + 1))
        return new_trial_numbers, sorted_objective_values
    else:
        # Generate trial numbers in their original order
        trial_numbers = list(range(1, len(objective_values) + 1))
        return trial_numbers, objective_values

# Function to collect all objective values from completed trials in a study
#def collect_objective_values(study):
#    return [trial.value for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None]


# Function to extract the best (minimum) loss value from each study
def get_best_loss_value_and_optimizer(study):
    best_trial = None
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None:
            if best_trial is None or trial.value < best_trial.value:
                best_trial = trial
    if best_trial:
        # Assuming the optimizer name is stored in trial.params under 'optimizer'
        optimizer_name = best_trial.params.get('optimizer', 'Unknown Optimizer')
        return best_trial.value, optimizer_name
    return None, None

# Function to calculate the minimum points for the Pareto front (minimization)
def calculate_min_points(trial_numbers, objective_values):
    min_points_trial = []
    min_points_objective = []
    
    current_min = float('inf')  # Start with a very high value
    
    for t, obj in zip(trial_numbers, objective_values):
        if obj < current_min:
            current_min = obj
        min_points_trial.append(t)
        min_points_objective.append(current_min)
    
    return min_points_trial, min_points_objective

# List of studies and storage URLs
study_infos = [
    {"name": 'distributed-FDN-Optuna', "url": "sqlite:///Encoder-Decoder-FDN-Optuna/distributed-FDN-Optuna.db"},
    {"name": 'distributed-DNF-Optuna', "url": "sqlite:///Encoder-Decoder-DNF-Optuna/distributed-DNF-Optuna.db"},
    {"name": 'distributed-CNN-Optuna', "url": "sqlite:///Encoder-Decoder-1DCNN-Optuna/distributed-CNN-Optuna.db"},
    {"name": 'distributed-TabNet-Optuna', "url": "sqlite:///Encoder-Decoder-TabNet2-Optuna/distributed-TabNet-Optuna.db"},
    {"name": 'distributed-VAE-Optuna', "url": "sqlite:///Encoder-Decoder-VAE-Optuna-NewParameters/distributed-VAE-Optuna.db"},
    {"name": 'distributed-XGBoost-Optuna', "url": "sqlite:///XGBoost-Optuna/distributed-XGBoost-Optuna.db"}
]

# Plot optimization history for all studies on one figure with log scale for y-axis
plt.figure(figsize=(12, 8))  # Updated figure size
for study_info in study_infos:
    storage = RDBStorage(url=study_info["url"])
    study = optuna.load_study(study_name=study_info["name"], storage=storage)
    
    # Collect sorted objective values and generate new trial numbers
    #trial_numbers, objective_values = collect_objective_values(study, sort_values=False)
    trial_numbers_sorted, objective_values_sorted = collect_objective_values(study, sort_values=True)

    if len(trial_numbers_sorted) > 0:
        # Remove 'distributed' and 'Optuna' from the study name for the label
        label_name = study_info["name"].replace('distributed-', '').replace('-Optuna', '')
        
        # Set dashed line style for VAE and XGBoost
        if 'VAE' in study_info["name"] or 'XGBoost' in study_info["name"]:
            plt.plot(trial_numbers_sorted, objective_values_sorted, label=label_name, lw=2.5, linestyle='--')
        else:
            plt.plot(trial_numbers_sorted, objective_values_sorted, label=label_name, lw=2.5)
        
# Set y-axis to log scale
plt.yscale('log')

# Final plot settings for optimization history
plt.xlabel("Trials (Sorted Order)", fontsize=25)
plt.ylabel("Objective Value (Log Scale)", fontsize=25)
#plt.title("Optimization History for All Studies (Log Scale, Sorted High to Low)", fontsize=16)

# Increase the font size for the axis numbers (tick labels)
plt.tick_params(axis='both', which='major', labelsize=20, width=1.5)  # Adjusted tick width

# Increase the linewidth of the plot box (axes)
ax = plt.gca()  # Get current axis
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# Increase the font size for the axis numbers (tick labels)
plt.tick_params(axis='both', which='major', labelsize=20)  # Adjust `labelsize` as needed
plt.legend(loc="best", fontsize=18)
plt.xlim([0,130])
plt.tight_layout()
plt.savefig('optuna-history.png', dpi=300)
plt.show()

# 

# Plot optimization history for all studies on one figure with log scale for y-axis
for study_info in study_infos:
    storage = RDBStorage(url=study_info["url"])
    study = optuna.load_study(study_name=study_info["name"], storage=storage)
    
    plt.figure(figsize=(12, 8))  # Updated figure size
    # Collect sorted objective values and generate new trial numbers
    trial_numbers, objective_values = collect_objective_values(study, sort_values=False)
    #trial_numbers_sorted, objective_values_sorted = collect_objective_values(study, sort_values=True)

    if len(trial_numbers_sorted) > 0:
        # Remove 'distributed' and 'Optuna' from the study name for the label
        label_name = study_info["name"].replace('distributed-', '').replace('-Optuna', '')
        
        fig, ax = plt.subplots(figsize=(12, 8))

        plt.scatter(trial_numbers, objective_values, label=label_name, marker='s', lw=2.5)
        #optuna.visualization.matplotlib.plot_optimization_history(study, ax=ax)
        # Get the Figure from the Axes object
        #fig = ax.get_figure()        
        #fig.set_size_inches(12, 8)
        #fig = plt.gcf()

        # Calculate min points to plot the connecting line
        min_trial_numbers, min_objective_values = calculate_min_points(trial_numbers, objective_values)
        
        # Plot the line connecting minimum points
        plt.plot(min_trial_numbers, min_objective_values, color='red', label='Minimum Points Line', lw=2.5)

        # Remove background color (set face color to white)
        #ax.set_facecolor('white')  # Set the background color to white

        # Remove background color (set face color to white)
        #ax = fig.gca()  # Get the current axes
        #ax.set_facecolor('white')  # Set the background color to white
        
        # Move legend inside the plot box
        ax.legend(loc='upper right', frameon=False, fontsize=20)  # Adjust the position and style of the legend
        
        # Show the plot
        plt.tight_layout()  # Adjust layout for better fit                
        # Set y-axis to log scale
        plt.yscale('log')

        # Final plot settings for optimization history
        plt.xlabel("Trials", fontsize=40)
        plt.ylabel("Objective Value", fontsize=40)
        #plt.title("Optimization History for All Studies (Log Scale, Sorted High to Low)", fontsize=16)

        # Increase the font size for the axis numbers (tick labels)
        plt.tick_params(axis='both', which='major', labelsize=50, width=2)  # Adjusted tick width
        
        # Increase the linewidth of the plot box (axes)
        ax = plt.gca()  # Get current axis
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Increase the font size for the axis numbers (tick labels)
        plt.tick_params(axis='both', which='major', labelsize=50)  # Adjust `labelsize` as needed
        plt.legend(loc="upper right", fontsize=35)
        plt.xlim([0,130])
        plt.ylim([0.0,5e-1])
        plt.tight_layout()
        plt.savefig("optuna-history-"+study_info["name"]+".png", dpi=300)
        plt.show()

# %%

# Initialize a list to store trial information
convergence_rates = []

# Loop through each study and extract the relevant details
for study_info in study_infos:
    storage = RDBStorage(url=study_info["url"])
    study = optuna.load_study(study_name=study_info["name"], storage=storage)
    
    trial_losses = []
    for trial in study.trials:
        if trial.value is not None:
            trial_losses.append(trial.value)

    # Calculate the convergence rate based on the changes in trial losses
    convergence_rate = [(trial_losses[i-1] - trial_losses[i]) / trial_losses[i-1] * 100 for i in range(1, len(trial_losses))]

    avg_convergence_rate = sum(convergence_rate) / len(convergence_rate) if convergence_rate else None
    
    convergence_rates.append({
        "Model": study_info["name"],
        "Average Convergence Rate (%)": avg_convergence_rate
    })

# Convert the results to a pandas DataFrame and display it
df_convergence_rates = pd.DataFrame(convergence_rates)

# Display the DataFrame
print(df_convergence_rates)

# Initialize a list to store the best loss values and other columns for each study
best_loss_values = []

# Loop through each study and extract the relevant details
for study_info in study_infos:
    storage = RDBStorage(url=study_info["url"])
    study = optuna.load_study(study_name=study_info["name"], storage=storage)
    
    trial_losses = []
    for trial in study.trials:
        if trial.value is not None:
            trial_losses.append(trial.value)

    # Calculate the convergence rate based on the changes in trial losses
    convergence_rate = [(trial_losses[i-1] - trial_losses[i]) / trial_losses[i-1] * 100 for i in range(1, len(trial_losses))]

    avg_convergence_rate = sum(convergence_rate) / len(convergence_rate) if convergence_rate else None

    
    best_loss, best_optimizer = get_best_loss_value_and_optimizer(study)
    best_trial_duration = study.best_trial.duration.total_seconds()  # Get duration in seconds
    test_r2 = study.best_trial.user_attrs.get("test_r2", None)  # Get test_r2 if available
    
    best_loss_values.append({
        "Model": study_info["name"], 
        "Best Loss Value": best_loss,
        "Optimizer": best_optimizer,
        "Best Trial Duration": best_trial_duration,
        "Test R²": test_r2,  # Add test_r2
        "Avg. Convergence Rate": avg_convergence_rate  # Add test_r2
    })

# Convert the results to a pandas DataFrame and display it
df_best_losses = pd.DataFrame(best_loss_values)

df_best_losses["Best Trial Duration"] = df_best_losses["Best Trial Duration"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else x)
df_best_losses["Test R²"] = df_best_losses["Test R²"].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else x)
df_best_losses["Avg. Convergence Rate"] = df_best_losses["Avg. Convergence Rate"].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else x)

# Sort the DataFrame based on the 'Best Loss Value' column in ascending order
df_best_losses_sorted = df_best_losses.sort_values(by="Best Loss Value", ascending=True)

# Display the sorted DataFrame
print(df_best_losses_sorted)

# Convert to LaTeX table with bold column names
latex_table = df_best_losses_sorted.to_latex(index=False, 
                                      caption="Best Loss Values, Optimizers, Best Trial Duration, and Test R² for Each Model", 
                                      label="tab:best_loss_values", 
                                      column_format="lcccc",  # Update column format for new column
                                      header=True)

# Update the LaTeX table to make column names bold
latex_table = latex_table.replace("\\toprule", "\\toprule\n\\textbf{Model} & \\textbf{Best Loss Value} & \\textbf{Optimizer} & \\textbf{Best Trial Duration} & \\textbf{Test R²} \\\\")

print(latex_table)


# %%

# Create heatmap for all studies
objective_data = []
max_trials = 0

# First, find the maximum number of trials across all studies
# for study_info in study_infos:
#     storage = RDBStorage(url=study_info["url"])
#     study = optuna.load_study(study_name=study_info["name"], storage=storage)
#     values = collect_objective_values(study)
#     objective_data.append(values)
#     max_trials = max(max_trials, len(values))

for study_info in study_infos:
    storage = RDBStorage(url=study_info["url"])
    study = optuna.load_study(study_name=study_info["name"], storage=storage)
    _, values = collect_objective_values(study, sort_values=True)  # or False as needed
    objective_data.append(values)
    max_trials = max(max_trials, len(values))

# Pad the shorter lists with NaN so that all lists have the same length
padded_data = []
for values in objective_data:
    padded_values = values + [np.nan] * (max_trials - len(values))  # Pad with NaN
    padded_data.append(padded_values)

# Convert to a NumPy array for heatmap
padded_data = np.array(padded_data)
log_padded_data = np.log(padded_data + 1e-9)

# Create the heatmap
plt.figure(figsize=(15, 2))
sns.heatmap(log_padded_data, cmap="YlGnBu", xticklabels=True, yticklabels=[study["name"] for study in study_infos])
plt.xlabel("Trial Number")
plt.ylabel("Study")
plt.title("Heatmap of Objective Values")
plt.show()

# %%

# Create a boxplot for objective values of all studies
box_data = []
labels = []
for study_info in study_infos:
    storage = RDBStorage(url=study_info["url"])
    study = optuna.load_study(study_name=study_info["name"], storage=storage)
    values = collect_objective_values(study)
    box_data.append(values)
    labels.append(study_info["name"])

# Plot boxplot for distribution of objective values
plt.figure(figsize=(10, 6))
sns.boxplot(data=box_data)
plt.xticks(ticks=range(len(labels)), labels=labels)
plt.title("Distribution of Objective Values for Each Study")
plt.ylabel("Objective Value")
plt.show()


# %%

import matplotlib.pyplot as plt
from optuna.storages import RDBStorage
from optuna.importance import get_param_importances
import re

for study_info in study_infos:
    
    print(study_info)
    
    storage = RDBStorage(url=study_info["url"])
    study = optuna.load_study(study_name=study_info["name"], storage=storage)

    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) == 0:
        print(f"No completed trials for {study_info['name']}, skipping plot.")
        continue

    # Get importances
    importances = get_param_importances(study)
    print(importances)
    
    # Sort by importance descending
    params = list(importances.keys())
    values = list(importances.values())
    sorted_pairs = sorted(zip(values, params), reverse=True)
    sorted_values, sorted_params = zip(*sorted_pairs)
        
    # Make labels nicer: replace underscores with spaces, capitalize
    nice_labels = [p.replace('_', ' ').capitalize() for p in sorted_params]
        
    
    short_labels = []
    for p in sorted_params:
        p_lower = p.lower()
        p_clean = p_lower.replace('encoder', 'ENC').replace('decoder', 'DEC').replace('layer', 'L').replace('strength_feature_penalty (gamma)', 'feature reusage').replace('dec', 'DEC').replace('sparsity reg (lambda_sparse)', 'sparsity loss coeff.')
        p_clean = p_clean.replace('_', ' ')
        p_clean = re.sub(r'(\s+)',' ', p_clean).strip()  # remove extra spaces
        short_labels.append(p_clean)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    bars = ax.barh(short_labels, sorted_values, color='steelblue')
    ax.set_xlabel('Importance', color='black')
    #ax.set_ylabel('Hyperparameter', color='black')
    ax.tick_params(colors='black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.set_xlim([0,1])
    
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    
    # Make tick labels black & bold
    for label in ax.get_yticklabels():
        label.set_color('black')
        label.set_fontweight('bold')
    
    # Add importance values next to bars
    for bar, value in zip(bars, sorted_values):
        ax.text(
            bar.get_width() + 0.01,   # x position, slightly to the right of bar
            bar.get_y() + bar.get_height()/2,  # center vertically
            f"{value:.2f}",            # formatted value
            va='center', 
            ha='left', 
            color='black',
            fontweight='bold',
            fontsize=9
        )
    
    ax.invert_yaxis()  # highest importance on top
    plt.tight_layout()
    plt.savefig(f"hyperparam-importance_{study_info['name']}.png", dpi=300)
    plt.show()
    
# %%

# Loop through each study to extract best hyperparameters
for study_info in study_infos:
    storage = RDBStorage(url=study_info["url"])
    study = optuna.load_study(study_name=study_info["name"], storage=storage)

    # Extract the best hyperparameters
    best_params = study.best_trial.params
    
    # Include the study name in the output
    best_params_with_study = {"Study Name": study_info["name"]}
    best_params_with_study.update(best_params)
    
    ## Print the study name followed by its best hyperparameters
    print(f"Best hyperparameters for {study_info['name']}:")
    for param, value in best_params_with_study.items():
        print(f"  {param}: {value}")

# Initialize a list to store the best hyperparameters for each study
best_params_list = []

# Loop through each study to extract best hyperparameters
for study_info in study_infos:
    storage = RDBStorage(url=study_info["url"])
    study = optuna.load_study(study_name=study_info["name"], storage=storage)

    # Extract the best hyperparameters
    best_params = study.best_trial.params
    #best_params['Study Name'] = study_info["name"]  # Add study name to track
    best_params_list.append(best_params)  # Append best params to the list

# Convert the list of best parameters to a DataFrame
df_best_params = pd.DataFrame(best_params_list)

print(df_best_params)

# Convert DataFrame to a LaTeX table
latex_table = df_best_params.to_latex(index=False, float_format="%.3f")

# Print the LaTeX table
print(latex_table)

# %%

# Initialize a list to store the best hyperparameters for each study
best_params_list = []

# Loop through each study to extract best hyperparameters
for study_info in study_infos:
    storage = RDBStorage(url=study_info["url"])
    study = optuna.load_study(study_name=study_info["name"], storage=storage)

    # Extract the best hyperparameters
    best_params = study.best_trial.params

    # Include the study name in the parameters
    best_params_with_study = {
        "Study Name": study_info["name"],
        "latent_dim": best_params.get("latent_dim", "-"),
        "drop_out_rate": best_params.get("drop_out_rate", "-"),
        "learning_rate": best_params.get("learning_rate", "-"),
        "optimizer": best_params.get("optimizer", "-"),
        "batch_size": best_params.get("batch_size", "-"),
        "epochs": best_params.get("epochs", "-")
    }

    # Collect the additional hyperparameters into a single field
    additional_hyperparams = []
    for param, value in best_params.items():
        if param not in ["latent_dim", "drop_out_rate", "learning_rate", "optimizer", "batch_size", "epochs"]:
            additional_hyperparams.append(f"{param}={value}")
    
    # Join the additional hyperparameters as a single string
    best_params_with_study["Additional Hyperparameters"] = ", ".join(additional_hyperparams) if additional_hyperparams else "-"

    # Append the best parameters with study name to the list
    best_params_list.append(best_params_with_study)

# Convert the list of best parameters to a DataFrame
df_best_params = pd.DataFrame(best_params_list)

# Convert DataFrame to a LaTeX table
latex_table = df_best_params.to_latex(index=False, float_format="%.3g", column_format="lccccccc", escape=False)

# Print the LaTeX table
print(latex_table)

# %%

import optuna
import pandas as pd
from optuna.storages import RDBStorage
from scipy.stats import friedmanchisquare
import optuna.visualization.matplotlib as optuna_viz
import seaborn as sns

# Initialize an empty list to store performance metrics
performance_data = []

# Loop through each study and extract the best performance metric
for study_info in study_infos:
    storage = RDBStorage(url=study_info["url"])
    study = optuna.load_study(study_name=study_info["name"], storage=storage)

    df_trials = study.trials_dataframe()

    
    # Collect sorted objective values and generate new trial numbers
    new_trial_numbers, sorted_objective_values = collect_and_sort_objective_values(study)

    performance_data.append(sorted_objective_values)
    
    # Plot slice to show how the values of latent_dim changed during optimization
    #optuna_viz.plot_slice(study, params=["latent_dim"])
    #plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.style.use('default')
    # Plot the PDF of 'learning_rate' with log scale
    sns.histplot(df_trials['params_learning_rate'].dropna(), kde=True, color='blue')
    # Apply log scaling to the x-axis
    plt.xscale('log')
    # Posterior distribution of the learning_rate after optimization
    plt.title("Posterior Distribution of Learning Rate")
    plt.savefig("Posterior_dis_Learning_Rate_"+str(study_info["name"])+".png",dpi=300)
    plt.show()
    
# Convert the performance data into a DataFrame (rows: datasets/trials, columns: models)
df_performance = pd.DataFrame(performance_data).T  # Transpose to get the right shape

# Print performance data for verification
print(df_performance.iloc[0:100,:])

# Perform Friedman's test
stat, p_value = friedmanchisquare(*df_performance.iloc[0:100,:].values)

# Print the results
print(f"Friedman's test statistic: {stat}")
print(f"P-value: {p_value}")

# Check if the result is statistically significant at a 95% confidence level
alpha = 0.05
if p_value < alpha:
    print("The differences in model performance are statistically significant.")
else:
    print("The differences in model performance are not statistically significant.")