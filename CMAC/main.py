from cProfile import label
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
from cmac import DiscreteCMAC, ContinuousCMAC, RecurrentCMAC
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
    n_weights = 35
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
                                    n_weights=n_weights, gen_factor=gen, overlap=overlap, mse_threshold=0.003)

                mse, training_duration = discrete_model_buffer[model_key].train()
                metric_dict['overlap'] = overlap
                metric_dict['mse'] = mse
                metric_dict['time'] = training_duration

                discrete_metrics.save_accuracy(metric_dict)
    
    # Plot results
    print("\n\n")
    print('RESULTS FOR DISCRETE CMAC')  
    discrete_metrics.plot_accuracy()
    discrete_metrics.plot_training_duration()
    discrete_metrics.plot_surface_map_mse()
    print("--------------------------")
    print("Best Accuracy Score:")
    disc_gen_min_mse, disc_overlap_min_mse, _ = discrete_metrics.get_minimum_mse()
    print("--------------------------")
    print("Worst Accuracy Score:")
    _,_,_ = discrete_metrics.get_maximum_mse()
    print("--------------------------")
    print("Best Time Score:")
    gen_min_time,_,_ = discrete_metrics.get_minimum_training_time()
    print("--------------------------")
    print("Worst Time Score:")
    _,_,_ = discrete_metrics.get_maximum_traning_time()
    discrete_metrics.plot_accuracy_accross_overlaps(n_weights, disc_gen_min_mse)
    discrete_metrics.plot_training_time_accross_overlaps(n_weights, gen_min_time)
    print("\n\n")

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
                                    n_weights=n_weights, gen_factor=gen, overlap=overlap, mse_threshold=0.003)

                mse, training_duration = continuous_model_buffer[model_key].train()
                metric_dict['overlap'] = overlap
                metric_dict['mse'] = mse
                metric_dict['time'] = training_duration

                continuous_metrics.save_accuracy(metric_dict)

    # Plot results
    print("\n\n")
    print('RESULTS FOR CONTINUOUS CMAC')  
    continuous_metrics.plot_accuracy()
    continuous_metrics.plot_training_duration()
    continuous_metrics.plot_surface_map_mse()
    print("--------------------------")
    print("Best Accuracy Score:")
    cont_gen_min_mse, cont_overlap_min_mse, _ = continuous_metrics.get_minimum_mse()
    print("--------------------------")
    print("Worst Accuracy Score:")
    _,_,_ = continuous_metrics.get_maximum_mse()
    print("--------------------------")
    print("Best Time Score:")
    gen_min_time,_,_ = continuous_metrics.get_minimum_training_time()
    print("--------------------------")
    print("Worst Time Score:")
    _,_,_ = continuous_metrics.get_maximum_traning_time()
    continuous_metrics.plot_accuracy_accross_overlaps(n_weights, cont_gen_min_mse)
    continuous_metrics.plot_training_time_accross_overlaps(n_weights, gen_min_time)
    print("\n\n")

    sample_points = np.linspace(0, 2*np.pi, 100)

    discrete_predicted = []
    continuous_predicted =[]
    real_values = []
    for sample in list(sample_points):
        real_values.append(training_function(sample))
        discrete_predicted.append(discrete_model_buffer[(n_weights, disc_gen_min_mse, disc_overlap_min_mse)].predict(sample))
        continuous_predicted.append(continuous_model_buffer[(n_weights, cont_gen_min_mse, cont_overlap_min_mse)].predict(sample))

    fig = plt.figure()        
    sns.set_style('darkgrid')
    fig.suptitle('Continuous CMAC, gen factor=' + str(cont_gen_min_mse) + ',overlap=' + str(cont_overlap_min_mse), fontsize=20)
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=16)
    plt.plot(sample_points, real_values, label='real values')
    plt.plot(sample_points, continuous_predicted, label='discrete cmac prediction')
    fig.legend()


    fig = plt.figure()        
    sns.set_style('darkgrid')
    fig.suptitle('Discrete CMAC, gen factor=' + str(disc_gen_min_mse) + ',overlap=' + str(disc_overlap_min_mse), fontsize=20)
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=16)
    plt.plot(sample_points, real_values, label='real values')
    plt.plot(sample_points, discrete_predicted, label='discrete cmac prediction')
    fig.legend()

    plt.show() 

if __name__ == "__main__":
    main()

    # sample_points = np.linspace(0, 2*np.pi, 100)
    # # Training data
    # training_sp = sample_points[:75]
    # # Test data
    # test_sp = sample_points[75:]

    # recurrent_cmac_model = RecurrentCMAC(train_samples=training_sp, test_samples=test_sp, train_function=training_function,
    #                                 n_weights=35, gen_factor=5, overlap=2,min_y=-1, max_y=1, mse_threshold=0.003)

    # mse = recurrent_cmac_model.train()
    # print('mes: ', mse)