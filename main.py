import numpy as np
import random
from load import loadData

from numpy.lib.function_base import gradient

"""
what we have:
- costFunction
- network struct
- evaluate()

next steps:
- backprop() -> how to change weights & biases for single example
- train() -> average the backprop of a mini batch of examples, shift weights and biases accordingly, prepare another mini batch, repeat for max iter

"""

class Network:
    def __init__(self):
        self.network = {}
        self.initNetwork()
    
    # Helpers
    def sigmoid(self, z):
        return (1 + np.exp(-z))**-1

    def sigmoid_deriv(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def initNetwork(self):
        # initiate weights
        l_weights = np.random.randn(3,2)
        l1_weights = np.random.randn(2,3)
        l2_weights = np.random.randn(3,4)

        self.network['weights'] = []
        self.network['weights'].append(l_weights)
        self.network['weights'].append(l1_weights)
        self.network['weights'].append(l2_weights)

        # initiate biases
        l_biases = np.random.randn(3) / 10
        l1_biases = np.random.randn(2) / 10
        l2_biases = np.random.randn(3) / 10

        self.network['biases'] = []
        self.network['biases'].append(l_biases)
        self.network['biases'].append(l1_biases)
        self.network['biases'].append(l2_biases)
    
    def evaluate(self, inpu):
        act = np.copy(inpu)

        for l in range(2, -1, -1):            
            curWeight = self.network["weights"][l]
            curBias = self.network["biases"][l]

            z = np.add( np.matmul(act, np.transpose(curWeight)), curBias)
            act = self.sigmoid(z)

        # print(act)
            
            
        return act
    
    def cost(self, groundTruth, output):
        return np.sum(np.square(np.subtract(groundTruth, output)))
    
    def backprop(self, input, groundTruth):
        act = np.copy(input)
        activations = [input]
        z_vals = []
        gradients = {"activations": [], "weights": [], "biases": []}        

        # create placeholder for gradient data, calc activation
        for l in range(2, -1, -1):            
            curWeight = self.network["weights"][l]
            curBias = self.network["biases"][l]

            z = np.add( np.matmul(act, np.transpose(curWeight)), curBias)
            act = self.sigmoid(z)
            activations.insert(0, act)
            z_vals.insert(0, z)

            gradients["activations"].insert(0, np.zeros_like(act))
            gradients["weights"].insert(0, np.zeros_like(curWeight))
            gradients["biases"].insert(0, np.zeros_like(curBias))
        
        # backprop algo bs
        # activations = [L, L-1, L-2, input]

        for n in range(len(activations[0])):
            dcda = 2 * (activations[0][n] - groundTruth[n])
            gradients["activations"][0][n] = dcda
        

        for l in range(3):
            for n in range(len(activations[l])):
                # Change in cost to change in weight
                dcda = gradients["activations"][l][n]
                dadz = self.sigmoid_deriv(z_vals[l][n])

                # iterate through weights connected to node
                for n2 in range(len(activations[l+1])):
                    dzdw = activations[l+1][n2]
                    dcdw = dcda * dadz * dzdw

                    gradients["weights"][l][n][n2] = dcdw

                    if(l != 2):
                        dcda_next_comp = dcda * dadz* self.network["weights"][l][n][n2]
                        gradients["activations"][l+1][n2] += dcda_next_comp
                        

                # Change in cost to change in biases
                dzdb = 1
                dcdb = dcda * dadz * dzdb

                gradients["biases"][l][n] = dcdb
        
        return gradients["weights"], gradients["biases"]

    def train(self, epochs, batchSize, learningRate):
        for i in range(epochs):
            inputs, groundTruths = loadData(batchSize)

            # Creating placeholder sum arrays
            l1 = np.zeros((3,2))
            l2 = np.zeros((2,3))
            l3 = np.zeros((3,4))

            mean_weight_grad = [l1, l2, l3]

            l1 = np.zeros(3)
            l2 = np.zeros(2)
            l3 = np.zeros(3)

            mean_bias_grad = [l1, l2, l3]

            # Calculating average gradients
            for j in range(batchSize):
                cur_weights, cur_biases = self.backprop(inputs[j], groundTruths[j])
                
                for l in range(3):
                    mean_weight_grad[l] = np.add(mean_weight_grad[l], cur_weights[l]) /batchSize 
                    mean_bias_grad[l] = np.add(mean_bias_grad[l], cur_biases[l]) / batchSize

            # Apply gradients
            for l in range(3):
                self.network["weights"][l] = np.subtract(self.network["weights"][l], mean_weight_grad[l]) * learningRate
                self.network["biases"][l] = np.subtract(self.network["biases"][l], mean_bias_grad[l]) * learningRate

    def test(self):
        inputs, groundTruths = loadData(30, "iris_test.csv")
        successCount = 0

        for i in range(len(inputs)):
            result = self.evaluate(inputs[i])

            # print(inputs[i], groundTruths[i], result)

            if np.argmax(result) == np.argmax(groundTruths[i]):
                successCount += 1

        return successCount/30

def test():
    nt = Network()

    print(nt.test())

    nt.train(2000, 40, 1)
    
    print(nt.test())
    
    # print(nt.network)
    # act = nt.evaluate([6.4,2.8,5.6,2.2])
    # print(act)
    # print(nt.cost(
    #     [0, 0, 1],
    #     act
    # ))

    # gradient = nt.backprop([6.4,2.8,5.6,2.2], [0, 0, 1])

    # print(gradient)

if __name__ == "__main__":
    test()