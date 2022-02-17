### DISCRETE CMAC ###

# Train on 1-D function
# * Tests effect of overlap on:
#   - Generalization
#   - Time to Converge

# Conditions:
# * 35 weights
# * Sample function at 100 evenly spaced points (70 for training and 30 for testing)
# NOTE: Report accuracy of CMAC only with the 30 test points

from random import sample
import numpy as np
import time
import math
import seaborn as sns
from tqdm import tqdm
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib import colors
from collections import defaultdict
from cmac import DiscreteCMAC, ContinuousCMAC
from metrics import Metrics



def training_function(x):
    return np.sin(x)

def main():
    
    sample_points = np.linspace(0, 2*np.pi, 100)
    # Split the data
    np.random.shuffle(sample_points)
    # Training data
    training_sp = sample_points[:70]
    # Test data
    test_sp = sample_points[70:]

    # Hyper-parameters
    n_weights = 3
    generalization_factors = np.linspace(1, n_weights, n_weights, dtype=int)
    alpha = 1

    metric_dict = {}
    metric_dict['n_weights'] = n_weights

    # Training of discrete models
    discrete_metrics = Metrics()
    discrete_model_buffer = {}
    for gen in tqdm(range(1,n_weights+1)):
        metric_dict['gen_factor'] = gen
        n_spare_weights = n_weights - gen
        spare_overlaps = list(filter(lambda x: (n_spare_weights % x == 0),
                                 np.linspace(1, n_spare_weights, n_spare_weights, dtype=int)))
        for s_over in spare_overlaps:
            overlap = gen - s_over
            model_key = (n_weights, gen, overlap)
            if gen >= s_over:
                discrete_model_buffer[model_key] = \
                    DiscreteCMAC(train_samples=training_sp, test_samples=test_sp, train_function=training_function,
                                    n_weights=n_weights, gen_factor=gen, overlap=overlap)

                mse, training_duration = discrete_model_buffer[model_key].train()
                metric_dict['overlap'] = overlap
                metric_dict['mse'] = mse
                metric_dict['time'] = training_duration

                discrete_metrics.save_accuracy(metric_dict)
        
    discrete_metrics.get_minimum_mse()
    discrete_metrics.plot_accuracy()
    discrete_metrics.plot_training_duration()

    # Training for continuous models
    continuous_metrics = Metrics()
    continuous_model_buffer = {}
    possible_s_over = [1, 2, 3, 4, 5, 6]
    for gen in tqdm(range(1,n_weights+1)):
        metric_dict['gen_factor'] = gen
        n_spare_weights = n_weights - gen
        spare_overlaps = list(filter(lambda x: (n_spare_weights % x == 0),
                                 np.linspace(1, n_spare_weights, n_spare_weights, dtype=int)))
        for s_over in spare_overlaps:
            overlap = gen - s_over
            if gen >= s_over and overlap !=0 and s_over in possible_s_over:
                model_key = (n_weights, gen, overlap)
                continuous_model_buffer[model_key] = \
                    ContinuousCMAC(train_samples=training_sp, test_samples=test_sp, train_function=training_function,
                                    n_weights=n_weights, gen_factor=gen, overlap=overlap)

                mse, training_duration = continuous_model_buffer[model_key].train()
                metric_dict['mse'] = mse
                metric_dict['time'] = training_duration

                continuous_metrics.save_accuracy(metric_dict)

    continuous_metrics.plot_accuracy()
    continuous_metrics.plot_training_duration()
    continuous_metrics.plot_surface_map_mse()

    plt.show() 

if __name__ == "__main__":
    main()