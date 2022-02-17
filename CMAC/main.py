### DISCRETE CMAC ###

# Train on 1-D function
# * Tests effect of overlap on:
#   - Generalization
#   - Time to Converge

# Conditions:
# * 35 weights
# * Sample function at 100 evenly spaced points (70 for training and 30 for testing)
# NOTE: Report accuracy of CMAC only with the 30 test points

from random import sample, shuffle
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

COLORS = list(colors.cnames.keys())
shuffle(COLORS)

class Metrics:
    def __init__(self) -> None:
        self.gen_facts = np.arange(0,35,1)
        self.overlaps = np.arange(0,35,1)
        self.gen_facts, self.overlaps = np.meshgrid(self.gen_facts, self.overlaps)
        self.mse = np.zeros((35,35))
        self.accuracies = defaultdict(lambda: defaultdict(dict))
        self.training_times = defaultdict(lambda: defaultdict(dict))
    
    def save_accuracy(self, metric):
        n_weights = metric['n_weights']
        gen_factor = metric['gen_factor']
        overlap = metric['overlap']
        self.accuracies[n_weights][gen_factor][overlap] = metric['mse']
        self.training_times[n_weights][gen_factor][overlap] = metric['time']
        self.mse[gen_factor, overlap] = metric['mse']

    def plot_accuracy(self):
        plt.figure()        
        sns.set_style('darkgrid')
        min_accuracy = []
        min_overlaps = []
        min_mse = []
        gen_factors = []
        for n_weights, weight_values in self.accuracies.items():
            for gen_factor, gen_f_values in weight_values.items():
                overlaps = []
                mse = []
                gen_factors.append(gen_factor)
                for overlap, mse_value in gen_f_values.items():
                    overlaps.append(overlap)
                    mse.append(mse_value)
                min_mse.append(min(mse))
                min_mse_idx = mse.index(min_mse[-1])
                min_accuracy.append(min_mse)
                min_overlaps.append(overlaps[min_mse_idx])
        plt.bar(gen_factors, min_mse)
    
    def plot_training_duration(self):
        plt.figure()        
        sns.set_style('darkgrid')
        min_time = []
        min_trainig_time = []
        min_overlaps = []
        gen_factors = []
        for n_weights, weight_values in self.training_times.items():
            for gen_factor, gen_f_values in weight_values.items():
                overlaps = []
                time = []
                gen_factors.append(gen_factor)
                for overlap, time_value in gen_f_values.items():
                    overlaps.append(overlap)
                    time.append(time_value)
                min_time.append(min(time))
                min_time_idx = time.index(min_time[-1])
                min_trainig_time.append(min_time)
                min_overlaps.append(overlaps[min_time_idx])
        plt.bar(gen_factors, min_time)

    def plot_accuracy_heatmap(self):
        ax = sns.heatmap(self.mse, linewidth=0.5)

    def plot_surface_map_mse(self):
        plt.figure()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(self.gen_facts, self.overlaps, self.mse, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)   
    
    def get_minimum_mse(self):
        min_accuracy = []
        min_overlaps = []
        min_mse = []
        gen_factors = []
        for n_weights, weight_values in self.accuracies.items():
            for gen_factor, gen_f_values in weight_values.items():
                overlaps = []
                mse = []
                gen_factors.append(gen_factor)
                for overlap, mse_value in gen_f_values.items():
                    overlaps.append(overlap)
                    mse.append(mse_value)
                min_mse.append(min(mse))
                min_mse_idx = mse.index(min_mse[-1])
                min_accuracy.append(min_mse)
                min_overlaps.append(overlaps[min_mse_idx])
        
        global_mse_min = min(min_mse)
        global_mse_min_idx = mse.index(global_mse_min)
        global_overlap_min = min_overlaps[global_mse_min_idx]
        global_gen_min = gen_factors[global_mse_min_idx]

        print('\tGen Factor: ', global_gen_min)
        print('\tOverlap: {}'.format(global_overlap_min))
        print('\tMSE: {}'.format(global_mse_min))

        return global_gen_min, global_overlap_min, global_mse_min
    
    def get_maximum_mse(self):
        max_accuracy = []
        max_overlaps = []
        max_mse = []
        gen_factors = []
        for n_weights, weight_values in self.accuracies.items():
            for gen_factor, gen_f_values in weight_values.items():
                overlaps = []
                mse = []
                gen_factors.append(gen_factor)
                for overlap, mse_value in gen_f_values.items():
                    overlaps.append(overlap)
                    mse.append(mse_value)
                max_mse.append(max(mse))
                max_mse_idx = mse.index(max_mse[-1])
                max_accuracy.append(max_mse)
                max_overlaps.append(overlaps[max_mse_idx])
        
        global_mse_max = max(max_mse)
        global_mse_max_idx = mse.index(global_mse_max)
        global_overlap_max = max_overlaps[global_mse_max_idx]
        global_gen_max = gen_factors[global_mse_max_idx]

        print('\tGen Factor: ', global_gen_max)
        print('\tOverlap: {}'.format(global_overlap_max))
        print('\tMSE: {}'.format(global_mse_max))

        return global_gen_max, global_overlap_max, global_mse_max

    def get_minimum_training_time(self):
        min_training_time = []
        min_overlaps = []
        min_time = []
        gen_factors = []
        for n_weights, weight_values in self.accuracies.items():
            for gen_factor, gen_f_values in weight_values.items():
                overlaps = []
                time = []
                gen_factors.append(gen_factor)
                for overlap, mse_value in gen_f_values.items():
                    overlaps.append(overlap)
                    time.append(mse_value)
                min_time.append(min(time))
                min_time_idx = time.index(min_time[-1])
                min_training_time.append(min_time)
                min_overlaps.append(overlaps[min_time_idx])
        
        global_time_min = min(min_time)
        global_time_min_idx = time.index(global_time_min)
        global_overlap_min = min_overlaps[global_time_min_idx]
        global_gen_min = gen_factors[global_time_min_idx]

        print('\tGen Factor: ', global_gen_min)
        print('\tOverlap: {}'.format(global_overlap_min))
        print('\tTime: {} seconds'.format(global_time_min))
    
        return global_gen_min, global_overlap_min, global_time_min

    def get_maximum_traning_time(self):
        min_training_time = []
        min_overlaps = []
        min_time = []
        gen_factors = []
        for n_weights, weight_values in self.accuracies.items():
            for gen_factor, gen_f_values in weight_values.items():
                overlaps = []
                time = []
                gen_factors.append(gen_factor)
                for overlap, mse_value in gen_f_values.items():
                    overlaps.append(overlap)
                    time.append(mse_value)
                min_time.append(min(time))
                min_time_idx = time.index(min_time[-1])
                min_training_time.append(min_time)
                min_overlaps.append(overlaps[min_time_idx])
        
        global_time_min = min(min_time)
        global_time_min_idx = time.index(global_time_min)
        global_overlap_min = min_overlaps[global_time_min_idx]
        global_gen_min = gen_factors[global_time_min_idx]

        print('\tGen Factor: ', global_gen_min)
        print('\tOverlap: {}'.format(global_overlap_min))
        print('\tTime: {} seconds'.format(global_time_min))
    
        return global_gen_min, global_overlap_min, global_time_min



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