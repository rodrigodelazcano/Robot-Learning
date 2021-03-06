from difflib import SequenceMatcher
import numpy as np
import math
import time


class CMAC:
    def __init__(self, train_samples, test_samples, train_function, 
                 n_weights, gen_factor, overlap, n_epochs=10000, alpha=1, mse_threshold=0.001):
        self.n_weights = n_weights
        self.gen_factor = gen_factor
        self.overlap = overlap
        self.s_overlap = gen_factor - overlap
        self.n_epochs = n_epochs
        self.mse_threshold = mse_threshold
        self.alpha = alpha

        self.train_samples = train_samples
        self.test_samples = test_samples
        self.training_function = train_function

        self.weights = np.ones(self.n_weights)

        all_samples = np.concatenate((train_samples, test_samples), axis=0)
        self.min_sp = np.min(all_samples)
        self.max_sp = np.max(all_samples)

        self._initialize_association_matrix()
    
    def _initialize_association_matrix(self):
        
        self.quantized_input = int(((self.n_weights-self.gen_factor)/self.s_overlap) + 1)
        self.quantization_step = (self.max_sp-self.min_sp)/self.quantized_input
        self.assoc_matrix = np.zeros((self.quantized_input, self.n_weights))
    
    def train(self):
        epoch = 0
        mse = np.inf

        starting_time = time.perf_counter()
        while (epoch < self.n_epochs) and (mse > self.mse_threshold):
            # Re-shuffle training samples
            np.random.shuffle(self.train_samples)
            for sample in self.train_samples:
                association_idx = math.floor((sample+abs(self.min_sp))/self.quantization_step)
                if association_idx == self.quantized_input:
                    association_idx -= 1
                association_vector = self.assoc_matrix[association_idx,:]
                y_pred = np.matmul(self.weights, association_vector)
                y = self.training_function(sample)
                error = y - y_pred
                correction = association_vector * (self.alpha * error / self.gen_factor)
                self.weights += correction
            epoch += 1

            # Check for early stopping every 5 epochs
            if epoch % 5 == 0:
                # Compute mean square error for testing samples
                cumulative_square_error = 0
                for sample in self.test_samples:
                    association_idx = math.floor((sample+abs(self.min_sp))/self.quantization_step)
                    if association_idx == self.quantized_input:
                        association_idx -= 1
                    association_vector = self.assoc_matrix[association_idx,:]
                    y_pred = np.matmul(self.weights, association_vector)
                    y = self.training_function(sample)
                    cumulative_square_error += math.pow((y - y_pred), 2)
                
                mse = cumulative_square_error/len(self.test_samples)
        ending_time = time.perf_counter()
        training_duration = ending_time - starting_time

        return mse, training_duration
    
    def predict(self, sample):
        association_idx = math.floor((sample+abs(self.min_sp))/self.quantization_step)
        if association_idx == self.quantized_input:
            association_idx -= 1
        association_vector = self.assoc_matrix[association_idx,:]    
        y_pred = np.matmul(self.weights, association_vector)

        return y_pred

class DiscreteCMAC(CMAC):
    def __init__(self, train_samples, test_samples, train_function, 
                 n_weights, gen_factor, overlap, n_epochs=10000, alpha=1, mse_threshold=0.001) -> None:
        
        super(DiscreteCMAC, self).__init__(train_samples, test_samples, train_function, 
                 n_weights, gen_factor, overlap, n_epochs, alpha, mse_threshold)
    
    def _initialize_association_matrix(self):
        super()._initialize_association_matrix()
        for idx in range(self.assoc_matrix.shape[0]):
            self.assoc_matrix[idx,idx*self.s_overlap:idx*self.s_overlap+self.gen_factor] = 1


class ContinuousCMAC(CMAC):
    def __init__(self, train_samples, test_samples, train_function, 
                 n_weights, gen_factor, overlap, n_epochs=10000, alpha=1, mse_threshold=0.001) -> None:       
        self.possible_s_over = [1, 2, 3, 4, 5, 6]

        super(ContinuousCMAC, self).__init__(train_samples, test_samples, train_function, 
                 n_weights, gen_factor, overlap, n_epochs, alpha, mse_threshold)
    
    def _initialize_association_matrix(self):
        super()._initialize_association_matrix()
        assert(self.s_overlap in self.possible_s_over), "Overlap provided can't have that value"
        for idx in range(self.assoc_matrix.shape[0]):
            if idx == 0:
                self.assoc_matrix[idx,idx*self.s_overlap:idx*self.s_overlap+self.gen_factor] = 1
            else:
                self.assoc_matrix[idx,idx*self.s_overlap:idx*self.s_overlap+self.overlap] = 1

                # Add proportional values
                if self.s_overlap == 1:
                    #left side values
                    self.assoc_matrix[idx,idx*self.s_overlap-1] = 0.2
                    #right side values
                    
                    self.assoc_matrix[idx,idx*self.s_overlap+self.overlap] = 0.8
                elif self.s_overlap == 2:
                    #left side values
                    self.assoc_matrix[idx,idx*self.s_overlap-1] = 0.8
                    self.assoc_matrix[idx,idx*self.s_overlap-2] = 0.2
                    #right side values
                    self.assoc_matrix[idx,idx*self.s_overlap+self.overlap] = 0.8
                    self.assoc_matrix[idx,idx*self.s_overlap+self.overlap+1] = 0.2
                elif self.s_overlap == 3:
                    #left side values
                    self.assoc_matrix[idx,idx*self.s_overlap-1] = 0.8
                    self.assoc_matrix[idx,idx*self.s_overlap-2] = 0.5
                    self.assoc_matrix[idx,idx*self.s_overlap-3] = 0.2
                    #right side values
                    self.assoc_matrix[idx,idx*self.s_overlap+self.overlap] = 0.8
                    self.assoc_matrix[idx,idx*self.s_overlap+self.overlap+1] = 0.5
                    self.assoc_matrix[idx,idx*self.s_overlap+self.overlap+2] = 0.2
                elif self.s_overlap == 4:
                    #left side values
                    self.assoc_matrix[idx,idx*self.s_overlap-1] = 0.9
                    self.assoc_matrix[idx,idx*self.s_overlap-2] = 0.8
                    self.assoc_matrix[idx,idx*self.s_overlap-3] = 0.2
                    self.assoc_matrix[idx,idx*self.s_overlap-4] = 0.1
                    #right side values
                    self.assoc_matrix[idx,idx*self.s_overlap+self.overlap] = 0.9
                    self.assoc_matrix[idx,idx*self.s_overlap+self.overlap+1] = 0.8
                    self.assoc_matrix[idx,idx*self.s_overlap+self.overlap+2] = 0.2
                    self.assoc_matrix[idx,idx*self.s_overlap+self.overlap+3] = 0.1
                elif self.s_overlap == 5:
                    #left side values
                    self.assoc_matrix[idx,idx*self.s_overlap-1] = 0.9
                    self.assoc_matrix[idx,idx*self.s_overlap-2] = 0.8
                    self.assoc_matrix[idx,idx*self.s_overlap-3] = 0.5
                    self.assoc_matrix[idx,idx*self.s_overlap-4] = 0.2
                    self.assoc_matrix[idx,idx*self.s_overlap-5] = 0.1
                    #right side values
                    self.assoc_matrix[idx,idx*self.s_overlap+self.overlap] = 0.9
                    self.assoc_matrix[idx,idx*self.s_overlap+self.overlap+1] = 0.8
                    self.assoc_matrix[idx,idx*self.s_overlap+self.overlap+2] = 0.5
                    self.assoc_matrix[idx,idx*self.s_overlap+self.overlap+3] = 0.2
                    self.assoc_matrix[idx,idx*self.s_overlap+self.overlap+4] = 0.1
                elif self.s_overlap == 6:
                    #left side values
                    self.assoc_matrix[idx,idx*self.s_overlap-1] = 0.9
                    self.assoc_matrix[idx,idx*self.s_overlap-2] = 0.8
                    self.assoc_matrix[idx,idx*self.s_overlap-3] = 0.6
                    self.assoc_matrix[idx,idx*self.s_overlap-4] = 0.4
                    self.assoc_matrix[idx,idx*self.s_overlap-5] = 0.2
                    self.assoc_matrix[idx,idx*self.s_overlap-6] = 0.1
                    #right side values
                    self.assoc_matrix[idx,idx*self.s_overlap+self.overlap] = 0.9
                    self.assoc_matrix[idx,idx*self.s_overlap+self.overlap+1] = 0.8
                    self.assoc_matrix[idx,idx*self.s_overlap+self.overlap+2] = 0.6
                    self.assoc_matrix[idx,idx*self.s_overlap+self.overlap+3] = 0.4
                    self.assoc_matrix[idx,idx*self.s_overlap+self.overlap+4] = 0.2
                    self.assoc_matrix[idx,idx*self.s_overlap+self.overlap+5] = 0.1

class RecurrentCMAC(CMAC):
    
    def __init__(self, train_samples, test_samples, train_function, 
                 n_weights, gen_factor, overlap, min_y, max_y, n_cells=3, n_epochs=10000, alpha=1, mse_threshold=0.001) -> None:       
        self.possible_s_over = [1, 2, 3, 4, 5, 6]
        self.min_y = min_y
        self.max_y = max_y
        super(RecurrentCMAC, self).__init__(train_samples, test_samples, train_function, 
                 n_weights, gen_factor, overlap, n_epochs, alpha, mse_threshold)
        
        self.n_cells = n_cells
        self.weights = []
        for cell in range(n_cells):
            if cell == 0:
                self.weights.append(np.ones(n_weights))
            else:
                self.weights.append(np.ones((n_weights, n_weights)))
        
        self._make_seq_samples(training_samples=train_samples, test_samples=test_samples)
        
    def _make_seq_samples(self, training_samples, test_samples):
        self.training_samples = []
        self.test_samples = []
        
        sequence = []
        for train_sp in training_samples:
            sequence.append(train_sp)

            if len(sequence) == self.n_cells:
                self.training_samples.append(np.array(sequence))
                sequence = []
        
        sequence = []
        for test_sp in test_samples:
            sequence.append(test_sp)

            if len(sequence) == self.n_cells:
                self.test_samples.append(np.array(sequence))
                sequence = []          

           
    def _initialize_association_matrix(self):
        self.quantized_input = int(((self.n_weights-self.gen_factor)/self.s_overlap) + 1)

        self.quantization_step_x = (self.max_sp-self.min_sp)/self.quantized_input
        self.assoc_matrix_x = np.zeros((self.quantized_input, self.n_weights))

        self.quantization_step_y = (self.max_y-self.min_y)/self.quantized_input
        self.assoc_matrix_y = np.zeros((self.quantized_input, self.n_weights))
        
        for idx in range(self.assoc_matrix_x.shape[0]):
            self.assoc_matrix_x[idx,idx*self.s_overlap:idx*self.s_overlap+self.gen_factor] = 1

        for idx in range(self.assoc_matrix_y.shape[0]):
            self.assoc_matrix_y[idx,idx*self.s_overlap:idx*self.s_overlap+self.gen_factor] = 1
    
    def train(self):
        epoch = 0
        mse = np.inf

        starting_time = time.perf_counter()
        while (epoch < self.n_epochs) and (mse > self.mse_threshold):
            # Re-shuffle training samples
            np.random.shuffle(self.training_samples)
            for sample in self.training_samples:
                y_pred = []
                for t in range(self.n_cells):
                    association_x_idx = math.floor((sample[t]+abs(self.min_sp))/self.quantization_step_x)
                    if association_x_idx == self.quantized_input:
                        association_x_idx -= 1
                    association_vector_x = self.assoc_matrix_x[association_x_idx,:]
                    # Predict with first cell. 1D input
                    if t == 0:
                        y_pred.append(np.matmul(self.weights[0], association_vector_x))
                        y = self.training_function(sample[t])
                        error = y - y_pred[-1]
                        correction = association_vector_x * (self.alpha * error / self.gen_factor)
                        self.weights[0] += correction
                        print('cell 0', y_pred)
                    # The rest of the cells are 2D. Compute association vector for previous cell output
                    else:
                        print(y_pred)
                        # prediction clipping
                        if y_pred[-1] > 1:
                            y_pred[-1] = 1
                        elif y_pred[-1] < -1:
                            y_pred[-1] = 1
                        # print(y_pred[-1])
                        # print(self.min_y)
                        # print(self.quantization_step_y)
                        association_y_idx = math.floor((y_pred[-1]+abs(self.min_y))/self.quantization_step_y)
                        if association_y_idx == self.quantized_input:
                            association_y_idx -= 1
                        association_vector_y = self.assoc_matrix_y[association_y_idx,:]
                        association_vector_x = association_vector_x.reshape(self.n_weights,1)
                        association_kernel = association_vector_x * association_vector_y

                        filtered_weights = np.multiply(self.weights[t], association_kernel)

                        y_pred.append(np.sum(filtered_weights))
                        y = self.training_function(sample[t])
                        error = y - y_pred[-1]
                        correction = association_kernel * (self.alpha * error / (2*self.gen_factor))
                        self.weights[t] += correction
            epoch += 1
            print(epoch)
            # Check for early stopping every 5 epochs
            if epoch % 5 == 0:
                # Compute mean square error for testing samples
                cumulative_square_error = 0
                for sample in self.test_samples:
                    y_pred = []
                    for t in range(self.n_cells):
                        association_x_idx = math.floor((sample[t]+abs(self.min_sp))/self.quantization_step_x)
                        if association_x_idx == self.quantized_input:
                            association_x_idx -= 1
                        association_vector_x = self.assoc_matrix_x[association_x_idx,:]
                        # Predict with first cell. 1D input
                        if t == 0:
                            y_pred.append(np.matmul(self.weights[0], association_vector_x))
                        # The rest of the cells are 2D. Compute association vector for previous cell output
                        else:
                            # prediction clipping
                            if y_pred[-1] > 1:
                                y_pred[-1] = 1
                            elif y_pred[-1] < -1:
                                y_pred[-1] = 1
                            association_y_idx = math.floor((y_pred[-1]+abs(self.min_y))/self.quantization_step_y)
                            if association_y_idx == self.quantized_input:
                                association_y_idx -= 1
                            association_vector_y = self.assoc_matrix_x[association_y_idx,:]
                            association_vector_x = association_vector_x.reshape(self.n_weights,1)
                            association_kernel = association_vector_x * association_vector_y

                            filtered_weights = np.multiply(self.weights[t], association_kernel)

                            y_pred.append(np.sum(filtered_weights))
                            if t == self.n_cells-1:
                                y = self.training_function(sample[t])
                                # print(y)
                                # print(y_pred[-1])
                                # cumulative_square_error += math.pow((y - y_pred[-1]), 2)
                print(epoch)
                # mse = cumulative_square_error/len(self.test_samples)

        return mse
