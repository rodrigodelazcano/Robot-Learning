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
    def save_accuracy(self, metric):
        n_weights = metric['n_weights']
        gen_factor = metric['gen_factor']
        overlap = metric['overlap']
        self.accuracies[n_weights][gen_factor][overlap] = metric['mse']

        
        self.mse[gen_factor, overlap] = metric['mse']
    def plot_accuracy(self):        
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
        plt.show()

    def plot_accuracy_surface(self):
        ax = sns.heatmap(self.mse, linewidth=0.5)      
        plt.show()
    
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

        print('BEST RESULT: ')
        print('\t Gen Factor: ', global_gen_min)
        print('\tOverlap: {}'.format(global_overlap_min))
        print('\tMSE: {}'.format(global_mse_min))



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
    n_weights = 35
    generalization_factors = np.linspace(1, n_weights, n_weights, dtype=int)
    alpha = 1
    
    # Overlaping in association matrix
    # Can we do overlapping
    # Quantization of input data
    # How much overlapping is possible
    # spare weights = n_weights - generalization_factor
    
    # 1 -> no overlapping
    # 2 -> 35/2 = 
    # Create a list of generalization factor's that make num weights divisible
    # overlap = generalization_factor - spare_overlap
    # quantized_input = n_weights/spare_overlap

    discrete_metrics = Metrics()
    accuracy_metric_dict = {}
    accuracy_metric_dict['n_weights'] = n_weights

    discrete_model_buffer = {}

    # Training of discrete models
    for gen in tqdm(range(1,n_weights+1)):
        accuracy_metric_dict['gen_factor'] = gen
        n_spare_weights = n_weights - gen
        spare_overlaps = list(filter(lambda x: (n_spare_weights % x == 0),
                                 np.linspace(1, n_spare_weights, n_spare_weights, dtype=int)))
        for s_over in spare_overlaps:
            overlap = gen - s_over
            model_key = (n_weights, gen, overlap)
            discrete_model_buffer[model_key] = \
                DiscreteCMAC(train_samples=training_sp, test_samples=test_sp, train_function=training_function,
                                n_weights=n_weights, gen_factor=gen, overlap=overlap)

            mse, training_duration = discrete_model_buffer[model_key].train()
    
    # Training for continuous models
    possible_s_over = [1, 2, 3, 4, 5, 6]
    for gen in tqdm(range(1,n_weights+1)):
        accuracy_metric_dict['gen_factor'] = gen
        n_spare_weights = n_weights - gen
        spare_overlaps = list(filter(lambda x: (n_spare_weights % x == 0),
                                 np.linspace(1, n_spare_weights, n_spare_weights, dtype=int)))
        for s_over in spare_overlaps:
            overlap = gen - s_over
            model_key = (n_weights, gen, overlap)
            discrete_model_buffer[model_key] = \
                ContinuousCMAC(train_samples=training_sp, test_samples=test_sp, train_function=training_function,
                                n_weights=n_weights, gen_factor=gen, overlap=overlap)

            mse = discrete_model_buffer[model_key].train()


    # Create association encoding map
    # for gen in tqdm(range(1,n_weights+1)):
    #     accuracy_metric_dict['gen_factor'] = gen
    #     n_spare_weights = n_weights - gen
    #     spare_overlaps = list(filter(lambda x: (n_spare_weights % x == 0),
    #                              np.linspace(1, n_spare_weights, n_spare_weights, dtype=int)))
        
    #     for s_over in spare_overlaps:
    #         # Reinitialize parameters
    #         weights = np.ones(n_weights)
    #         epoch = 0
    #         mse = np.inf
    #         if gen >= s_over:
    #             overlap = gen - s_over
    #             accuracy_metric_dict['overlap'] = overlap
    #             quantized_input = int((n_spare_weights/s_over) + 1)
    #             quantization_step = (max_sp-min_sp)/quantized_input
    #             assoc_matrix = np.zeros((quantized_input, n_weights))
    #             for idx in range(assoc_matrix.shape[0]):
    #                 assoc_matrix[idx,idx*s_over:idx*s_over+gen] = 1

    #             while (epoch < n_epochs) and (mse > e_threshold):
    #                 # Re-shuffle training samples
    #                 np.random.shuffle(training_sp)
    #                 for sample in training_sp:
    #                     association_idx = math.floor((sample+abs(min_sp))/quantization_step)
    #                     if association_idx == quantized_input:
    #                         association_idx -= 1
    #                     association_vector = assoc_matrix[association_idx,:]
    #                     y_pred = np.matmul(weights, association_vector)
    #                     y = training_function(sample)
    #                     error = y - y_pred
    #                     correction = association_vector * (alpha * error / gen)
    #                     weights += correction
    #                 epoch += 1
    #                     # print(correction)


    #                 # Check for early stopping every 5 epochs
    #                 if epoch % 5 == 0:
    #                     # Compute mean square error for testing samples
    #                     cumulative_square_error = 0
    #                     for sample in test_sp:
    #                         association_idx = math.floor((sample+abs(min_sp))/quantization_step)
    #                         if association_idx == quantized_input:
    #                             association_idx -= 1
    #                         association_vector = assoc_matrix[association_idx,:]
    #                         y_pred = np.matmul(weights, association_vector)
    #                         y = training_function(sample)
    #                         cumulative_square_error += math.pow((y - y_pred), 2)
                        
    #                     mse = cumulative_square_error/len(test_sp)
                
    #             accuracy_metric_dict['mse'] = mse

    #             discrete_metrics.save_accuracy(accuracy_metric_dict)
        
    #             print('--------------------------')
    #             print('GENERALIZATION FACTOR: ', gen)
    #             print('OVERLAP: ', overlap)
    #             # print('NUMBER OF QUANTILES: ', quantized_input)
    #             # print('NUMBER OF EPOCHS: ', epoch)
    #             print('FINAL ACCURACY: ', mse)

    # discrete_metrics.get_minimum_mse()
    # discrete_metrics.plot_accuracy()
    # time_start = time.perf_counter()

    # Training algorithm

    # print("Time to converge: {}".format(time.perf_counter()-time_start))

    # Print results

    ### CONTINUOUS ###

    # association vector elements have to sum up to generalization factor
    # overlap values 1

    # Parameters: gen_fact, overlap, weights

    # Create association matrix, quatize input as with discrete version.
    # The elements of the association vector for each quantile has to sum up to the generalization factor.
    # Overlap values equal to 1 the rest lower than 1 (proportions). Always satisfying the previous rule.

    

    
     

if __name__ == "__main__":
    main()