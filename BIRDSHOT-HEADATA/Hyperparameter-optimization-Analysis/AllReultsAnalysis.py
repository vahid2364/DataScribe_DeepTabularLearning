#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:05:59 2025

@author: attari.v
"""

import matplotlib.pyplot as plt
import optuna
import seaborn as sns
import numpy as np
import pandas as pd
from optuna.storages import RDBStorage
import re

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


if __name__ == "__main__":



    # List of studies and storage URLs
    study_infos = [
        {"name": 'distributed-FDN-Optuna', "url": "sqlite:///../Hyperparameter-optimization/Encoder-Decoder-FDN-Optuna-overcomplete-YS-SMAPE/distributed-FDN-Optuna.db", "label": "Yield Strength (YS)"},
        {"name": 'distributed-FDN-Optuna', "url": "sqlite:///../Hyperparameter-optimization/Encoder-Decoder-FDN-Optuna-overcomplete-UTS-SMAPE/distributed-FDN-Optuna.db", "label": "Ultimate Tensile Strength (UTS)"},
        {"name": 'distributed-FDN-Optuna', "url": "sqlite:///../Hyperparameter-optimization/Encoder-Decoder-FDN-Optuna-overcomplete-Elon T-SMAPE/distributed-FDN-Optuna.db", "label": "Elongation (T)"},
        {"name": 'distributed-FDN-Optuna', "url": "sqlite:///../Hyperparameter-optimization/Encoder-Decoder-FDN-Optuna-overcomplete-Hardness-SMAPE/distributed-FDN-Optuna.db", "label": "Hardness"},
        {"name": 'distributed-FDN-Optuna', "url": "sqlite:///../Hyperparameter-optimization/Encoder-Decoder-FDN-Optuna-overcomplete-Modulus-SMAPE/distributed-FDN-Optuna.db", "label": "Modulus"},
        {"name": 'distributed-FDN-Optuna', "url": "sqlite:///../Hyperparameter-optimization/Encoder-Decoder-FDN-Optuna-overcomplete-Avg HDYNHQSRatio-SMAPE/distributed-FDN-Optuna.db", "label": "Avg HDYN/HQS Ratio"},
        #{"name": 'distributed-FDN-Optuna', "url": "sqlite:///../Hyperparameter-optimization/Encoder-Decoder-FDN-Optuna-overcomplete-Depth of Penetration (mm) FE_Sim-SMAPE/distributed-FDN-Optuna.db", "label": "Depth of Penetration (FE Sim)"},
    ]
    
    
    # %% 

    # Plot optimization history for all studies on one figure with log scale for y-axis
    plt.figure(figsize=(12, 8))  # Updated figure size
    for study_info in study_infos:
        storage = RDBStorage(url=study_info["url"])
        study = optuna.load_study(study_name=study_info["name"], storage=storage)
        
        # Collect sorted objective values and generate new trial numbers
        #trial_numbers, objective_values = collect_objective_values(study, sort_values=False)
        trial_numbers_sorted, objective_values_sorted = collect_objective_values(study, sort_values=False)
    
        if len(trial_numbers_sorted) > 0:
            # Remove 'distributed' and 'Optuna' from the study name for the label
            label_name = study_info["label"]#.replace('distributed-', '').replace('-Optuna', '')
            
            # Set dashed line style for VAE and XGBoost
            if 'VAE' in study_info["label"] or 'XGBoost' in study_info["name"]:
                plt.plot(trial_numbers_sorted, objective_values_sorted, label=label_name, lw=2.5, linestyle='--')
            else:
                plt.plot(trial_numbers_sorted, objective_values_sorted, label=label_name, lw=2.5)
            
    # Set y-axis to log scale
    #plt.yscale('log')
    
    # Final plot settings for optimization history
    plt.xlabel("Trials", fontsize=25)
    plt.ylabel("Objective Value: SMAPE", fontsize=25)
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
    #plt.xlim([0,130])
    plt.tight_layout()
    plt.savefig('optuna-history.png', dpi=300)
    plt.show()
    
    # %% 
    
    # Plot optimization history for all studies on one figure with log scale for y-axis
    for study_info in study_infos:
        storage = RDBStorage(url=study_info["url"])
        study = optuna.load_study(study_name=study_info["name"], storage=storage)
        
        plt.figure(figsize=(12, 8))  # Updated figure size
        # Collect sorted objective values and generate new trial numbers
        trial_numbers, objective_values = collect_objective_values(study, sort_values=False)
        #trial_numbers_sorted, objective_values_sorted = collect_objective_values(study, sort_values=True)
    
        if len(trial_numbers_sorted) > 0:
            ## Remove 'distributed' and 'Optuna' from the study name for the label
            #label_name = study_info["label"]#.replace('distributed-', '').replace('-Optuna', '')
            label_name = re.sub(r'[<>:"/\\|?*]', '_', study_info["label"]).replace(" ", "_")

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
            ax.legend(loc='upper right', frameon=False, fontsize=30)  # Adjust the position and style of the legend
            
            # Show the plot
            plt.tight_layout()  # Adjust layout for better fit                

            # Set y-axis to log scale
            #plt.yscale('log')
    
            # Final plot settings for optimization history
            plt.xlabel("Trials", fontsize=30)
            plt.ylabel("Objective Value: SMAPE", fontsize=30)
            #plt.title("Optimization History for All Studies (Log Scale, Sorted High to Low)", fontsize=16)
    
            # Increase the font size for the axis numbers (tick labels)
            plt.tick_params(axis='both', which='major', labelsize=30, width=2)  # Adjusted tick width
            
            # Increase the linewidth of the plot box (axes)
            ax = plt.gca()  # Get current axis
            ax.spines['top'].set_linewidth(1.5)
            ax.spines['right'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            
            # Increase the font size for the axis numbers (tick labels)
            plt.tick_params(axis='both', which='major', labelsize=20, width=2)  # Adjust `labelsize` as needed
            plt.tick_params(axis='both', which='minor', labelsize=20, width=2)  # Adjust `labelsize` as needed
            plt.legend(loc="upper right", fontsize=30)
            #plt.xlim([0,130])
            #plt.ylim([15,100])
            plt.tight_layout()
            plt.savefig("optuna-history-"+label_name+".png", dpi=300)
            plt.show()
    
    pause

    
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
        test_r2 = study.best_trial.user_attrs.get("r2_test", None)  # Get test_r2 if available
        train_r2 = study.best_trial.user_attrs.get("r2_train", None)  # Get test_r2 if available
        smape_train = study.best_trial.user_attrs.get("smape_train", None)  # Get test_r2 if available
        smape_test  = study.best_trial.user_attrs.get("smape_test", None)  # Get test_r2 if available
        
        best_loss_values.append({
            "Model": study_info["label"], 
            #"Best Loss Value": round(best_loss, 1),
            "Optimizer": best_optimizer,
            "Best Trial Duration": round(best_trial_duration, 1),
            #"Train R²": round(train_r2, 1),  # Add train_r2
            #"Test R²": round(test_r2, 1),  # Add test_r2
            "Train SMAPE": round(smape_train, 1),  # Add smape_train
            "Test SMAPE": round(smape_test, 1),  # Add smape_test
            #"Avg. Convergence Rate": round(avg_convergence_rate, 1)  # Add avg_convergence_rate
        })
    
    # Convert the results to a pandas DataFrame and display it
    df_best_losses = pd.DataFrame(best_loss_values)
    
    df_best_losses["Best Trial Duration"] = df_best_losses["Best Trial Duration"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else x)
    #df_best_losses["Test R²"] = df_best_losses["Test R²"].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else x)
    #df_best_losses["Train R²"] = df_best_losses["Train R²"].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else x)
    #df_best_losses["Avg. Convergence Rate"] = df_best_losses["Avg. Convergence Rate"].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else x)
    
    # Sort the DataFrame based on the 'Best Loss Value' column in ascending order
    df_best_losses_sorted = df_best_losses.sort_values(by="Train SMAPE", ascending=True)
    
    # Display the sorted DataFrame
    print(df_best_losses_sorted)
    
    # Convert to LaTeX table with bold column names
    latex_table = df_best_losses_sorted.to_latex(index=False, 
                                          caption="Summary of Best Loss Values, Optimizers, Trial Durations, and Evaluation Metrics for Training and Testing Across Models", 
                                          label="tab:best_loss_values", 
                                          column_format="lcccccccc",  # Update column format for new column
                                          header=True,
                                          float_format="{:.1f}".format  # Apply floating point formatting
                                          )
    print('')
    
    # Update the LaTeX table to make column names bold
    latex_table = latex_table.replace("\\toprule", "\\toprule\n\\textbf{Model} & \\textbf{Optimizer} & \\textbf{Best Trial Duration} & \\textbf{Train SMAPE} & \\textbf{Test SMAPE}  \\\\")
    # Add the table* environment
    latex_table = latex_table.replace("\\begin{table}", "\\begin{table*}").replace("\\end{table}", "\\end{table*}")    
    
    print(latex_table)
    
        
    
    # %%
    
    # Create heatmap for all studies
    objective_data = []
    max_trials = 0
    
    # First, find the maximum number of trials across all studies
    for study_info in study_infos:
        storage = RDBStorage(url=study_info["url"])
        study = optuna.load_study(study_name=study_info["name"], storage=storage)
        values = collect_objective_values(study)  # Replace with your function to collect objective values
        objective_data.append(values)
        max_trials = max(max_trials, len(values))
    
    # Pad the shorter lists with NaN so that all lists have the same length
    # Pad the shorter lists with NaN so that all lists have the same length
    padded_data = [
        list(values) + [np.nan] * (max_trials - len(values)) for values in objective_data
    ]
    
    # Pad the shorter lists with NaN so that all lists have the same length
    padded_data = []
    for values in objective_data:
        # Convert tuple to list and pad with NaN
        padded_values = list(values) + [np.nan] * (max_trials - len(values))
        padded_data.append(padded_values)
    
    # Convert the padded data to a NumPy array
    padded_data_array = np.array(padded_data, dtype=np.float64)  # Ensure consistent numeric type
    
    # Create the heatmap
    plt.figure(figsize=(15, 2))
    sns.heatmap(padded_data_array, cmap="YlGnBu", xticklabels=True, yticklabels=[study["name"] for study in study_infos])
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
    import optuna.visualization.matplotlib as optuna_matplotlib
    
    for study_info in study_infos:
        storage = RDBStorage(url=study_info["url"])
        study = optuna.load_study(study_name=study_info["name"], storage=storage)
        
        # Check if the study has completed trials
        completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        
        if len(completed_trials) == 0:
            print(f"No completed trials for {study_info['name']}, skipping parallel coordinate plot.")
            continue
        
        # Plot parallel coordinate plot for the current study
        #fig = vis.plot_parallel_coordinate(study)
        # First, create the figure with your desired size
        
        # Plot the hyperparameter importance using the ax object
        fig = optuna_matplotlib.plot_param_importances(study)
        
        fig.figure.set_size_inches(6, 3)  # Set width and height
        fig.set_title('')  # Removes the title
        fig.figure.patch.set_facecolor('white')  # Set figure background to white
        fig.patch.set_facecolor('white')  # Set axes background to white
        # Make lines, ticks, and labels black
        fig.tick_params(colors='black')  # Ticks
        fig.xaxis.label.set_color('black')  # X-axis label
        fig.yaxis.label.set_color('black')  # Y-axis label
        fig.title.set_color('white')  # Title/ does not work
        
        # Set all spines (the lines around the plot) to black
        for spine in fig.spines.values():
            spine.set_edgecolor('black')
        
        # Change color of all text elements to black
        for label in fig.get_xticklabels() + fig.get_yticklabels():
            label.set_color('black')
        
        fig.legend(loc='lower right')
        
        plt.tight_layout()
        # Save the figure for each study (optional)
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