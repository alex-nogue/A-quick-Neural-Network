# Number of cores used: 4
# Free memory: 3.533Gb

# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h2o
h2o.init()

# Import data
url = "http://coursera.h2o.ai/cacao.882.csv"
data = h2o.import_file(url)
train, valid, test = data.split_frame([0.8, 0.1], seed = 10)

# Set the dependent and explanatory variables
y = 'Maker Location'
ignoreFields = ['Maker Location']
x = [i for i in data.names if i not in ignoreFields]

# Baseline model (1'15'' on my laptop)
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
baseline = H2ODeepLearningEstimator(seed = 10) 
%time baseline.train(x, y, train, validation_frame=valid)

# Model performance
baseline.mse(valid=True) # MSE = 0.263971569719431

# Save the model
model_path_baseline = h2o.save_model(model=baseline, path="/tmp/cacao", force=True)
print(model_path_baseline)

# Grid Search (5 minutes run)
from h2o.grid.grid_search import H2OGridSearch
grid = H2OGridSearch(H2ODeepLearningEstimator(seed = 10), 
                            hyper_params = {
                                'activation': ["RectifierWithDropout"],
                                'rate' : [0.005, 0.001],#, 0.01],
                                'l1': [0, 0.00001],#, 0.0001],
                                'l2': [0, 0.00001],#, 0.0001], 
                                'input_dropout_ratio' : [0, 0.1],
                                'hidden_dropout_ratios':[[0, 0], [0.2,0.2]]
                            })
%time grid.train(x, y, train, validation_frame=valid)

# Grid of models
gridperf1 = grid.get_grid(sort_by='mse', decreasing=False)
gridperf1.mse(valid=True)

# Best model
best_model = gridperf1.models[0]
best_model.mse(valid=True)

# Save the model
model_path_best = h2o.save_model(model=best_model, path="/tmp/cacao", force=True)
print(model_path_best)

# Load the model
saved_model = h2o.load_model(model_path)
