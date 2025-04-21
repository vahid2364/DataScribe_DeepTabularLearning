#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 10:35:34 2024

@author: attari.v
"""

import optuna
from optuna.distributions import UniformDistribution
import matplotlib.pyplot as plt

def objective(trial):
    x = trial.suggest_uniform('x', 0, 10)
    return x ** 2

study = optuna.create_study()

study.optimize(objective, n_trials=100)

print(study.best_params)
print(study.best_value)
print(study.best_trial)

# Plot optimization history
plt.figure(figsize=(10, 6))
ax = optuna.visualization.matplotlib.plot_optimization_history(study)
ax.set_facecolor('none')
for line in ax.get_lines():
    line.set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.tick_params(axis='both', colors='black')
ax.xaxis.label.set_color('black')
ax.yaxis.label.set_color('black')
plt.savefig('optimization_history-FDN.jpg')
plt.show()

 

# assert len(study.trials) == 0

# trial = optuna.trial.create_trial(
#     params={"x": 2.0},
#     distributions={"x": UniformDistribution(0, 10)},
#     value=4.0,
# )

# study.add_trial(trial)
# assert len(study.trials) == 1

# study.optimize(objective, n_trials=3)
# assert len(study.trials) == 4

# other_study = optuna.create_study()

# for trial in study.trials:
#     other_study.add_trial(trial)
# assert len(other_study.trials) == len(study.trials)

# other_study.optimize(objective, n_trials=2)
# assert len(other_study.trials) == len(study.trials) + 2