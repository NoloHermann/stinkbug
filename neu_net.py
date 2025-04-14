# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 10:33:50 2022

@author: herma
"""
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, learning_rate):
        #self.weights = np.array([np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn()])
        self.weights = np.array([np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )
        derror_dweights = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
            derror_dweights * self.learning_rate
        )
        
    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )

            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors
    
    
##############################################################################
f = open('mixBoth_log.txt','r')

mq2 = []
mq5 = []
mq6 = []
mq135 = []
ir = []
rgb = []

for line in f:
    x = line.split(",")
    mq2.append(int(x[0]))
    mq5.append(int(x[1]))
    mq6.append(int(x[2]))
    mq135.append(int(x[3]))
    ir.append(int(x[4]))
    rgb.append(int(x[5]))
    
f.close()
    
targets = [0]
for i in range(10):
    targets.append(0)
for i in range(63-10):
    targets.append(1)
for i in range(127-63):
    targets.append(0)
for i in range(189-127):
    targets.append(1)
for i in range(238-189):
    targets.append(0)
   
targets = np.array(targets)

input_vectors = []
for i in range(len(mq2)):
    #input_vectors.append([mq2[i], mq5[i], mq6[i], mq135[i], ir[i], rgb[i]])
    input_vectors.append([mq2[i], mq5[i], mq6[i], mq135[i]])
learning_rate = 0.1
neural_network = NeuralNetwork(learning_rate)
training_error = neural_network.train(input_vectors, targets, 1000000)

plt.plot(training_error)
plt.title("Neural network error")
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.show()

print(neural_network.weights)
print(neural_network.bias)

outp = []
for j in range(len(targets)):
    pred = neural_network.predict(input_vectors[j])
    outp.append(pred)
    
plt.plot(outp)
plt.plot(targets)
plt.title("Neural network output vs desired output")
plt.xlabel("Iterations")
plt.ylabel("Outputs")
plt.show()
    