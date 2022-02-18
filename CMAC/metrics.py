from random import shuffle
import numpy as np
import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib import colors
from collections import defaultdict


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
        fig = plt.figure()        
        sns.set_style('darkgrid')
        fig.suptitle('Minimum MSE Accuracy', fontsize=20)
        plt.xlabel('Generalization Factor', fontsize=18)
        plt.ylabel('MSE', fontsize=16)

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
        fig = plt.figure()        
        sns.set_style('darkgrid')
        fig.suptitle('Minimum Training Duration', fontsize=20)
        plt.xlabel('Generalization Factor', fontsize=18)
        plt.ylabel('Time (seconds)', fontsize=16)

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
    
    def plot_accuracy_accross_overlaps(self, n_weights, gen_factor):
        fig = plt.figure()        
        sns.set_style('darkgrid')
        fig.suptitle('MSE Accuracy for Generalization Factor ' + str(gen_factor), fontsize=20)
        plt.xlabel('Overlap', fontsize=18)
        plt.ylabel('MSE', fontsize=16)
        
        overlaps = []
        mse = []
        for overlap, mse_value in self.accuracies[n_weights][gen_factor].items():
            overlaps.append(overlap)
            mse.append(mse_value)
        plt.bar(overlaps, mse)

    def plot_training_time_accross_overlaps(self, n_weights, gen_factor):
        fig = plt.figure()        
        sns.set_style('darkgrid')
        fig.suptitle('Training Time for Generalization Factor ' + str(gen_factor), fontsize=20)
        plt.xlabel('Overlap', fontsize=18)
        plt.ylabel('Time (seconds)', fontsize=16)
        
        overlaps = []
        time = []
        for overlap, time_value in self.training_times[n_weights][gen_factor].items():
            overlaps.append(overlap)
            time.append(time_value)
        plt.bar(overlaps, time)

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
        global_mse_min_idx = min_mse.index(global_mse_min)
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
        global_mse_max_idx = max_mse.index(global_mse_max)
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
                min_training_time.append(min_time)
                min_overlaps.append(overlaps[min_time_idx])
        
        global_time_min = min(min_time)
        global_time_min_idx = min_time.index(global_time_min)
        global_overlap_min = min_overlaps[global_time_min_idx]
        global_gen_min = gen_factors[global_time_min_idx]

        print('\tGen Factor: ', global_gen_min)
        print('\tOverlap: {}'.format(global_overlap_min))
        print('\tTime: {} seconds'.format(global_time_min))
    
        return global_gen_min, global_overlap_min, global_time_min

    def get_maximum_traning_time(self):
        max_training_time = []
        max_overlaps = []
        max_time = []
        gen_factors = []
        for n_weights, weight_values in self.training_times.items():
            for gen_factor, gen_f_values in weight_values.items():
                overlaps = []
                time = []
                gen_factors.append(gen_factor)
                for overlap, time_value in gen_f_values.items():
                    overlaps.append(overlap)
                    time.append(time_value)
                max_time.append(max(time))
                max_time_idx = time.index(max_time[-1])
                max_training_time.append(max_time)
                max_overlaps.append(overlaps[max_time_idx])
        
        global_time_max = max(max_time)
        global_time_max_idx = max_time.index(global_time_max)
        global_overlap_max = max_overlaps[global_time_max_idx]
        global_gen_max = gen_factors[global_time_max_idx]

        print('\tGen Factor: ', global_gen_max)
        print('\tOverlap: {}'.format(global_overlap_max))
        print('\tTime: {} seconds'.format(global_time_max))
    
        return global_gen_max, global_overlap_max, global_time_max