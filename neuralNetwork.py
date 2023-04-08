import numpy as np
from load import loadData

class nNetwork:
    def __init__(self, format):
       self.format = format
       self.network = {}
       self.initNetwork()

    # Helpers. These are the activation functions. 
    def sigmoid(self, z):
        return (1 + np.exp(-z))**-1

    def sigmoid_deriv(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def initNetwork(self):
            # Add empty list to network
        self.network['weights'] = []
        self.network['biases'] = []

        # Initiating weight matricies, appending them to the list in the network dict.
        # Vals are random between -0.1 and 0.1
       
        for i in range(len(self.format)-1):
            weightToAppend = np.random.rand(self.format[i+1],self.format[i]) /5 -0.1
            self.network['weights'].append(weightToAppend)

            biasToAppend = np.random.rand(self.format[i+1]) /5 -0.1
            self.network['biases'].append(biasToAppend)
    
    def evaluate(self, inpu):
        #Copy the input array
        act = np.copy(inpu)

        for l in range(len(self.format)-1):            
            curWeight = self.network["weights"][l]
            curBias = self.network["biases"][l]
            z = np.add( np.matmul(curWeight,act), curBias)
            act = self.sigmoid(z)
            
        return act
    
    def cost(self, groundTruth, output):
        return np.sum(np.square(np.subtract(groundTruth, output))) 

    def backprop(self, input, groundTruth):
        act = np.copy(input)
        activations = [input]
        z_vals = []
        gradients = {"weights": [], "biases": []}        

        # create placeholder for gradient data, calc activation
        for l in range(len(self.format)-1):            
            curWeight = self.network["weights"][l]
            curBias = self.network["biases"][l]

            z = np.add( np.matmul( curWeight,act), curBias)

            act = self.sigmoid(z)
            activations.append(act)
            z_vals.append(z)

            gradients["weights"].append(np.zeros_like(curWeight))
            gradients["biases"].append(np.zeros_like(curBias))
        
        #Actual Algorithim
       
        gradients["biases"][-1] = 2*np.multiply(np.subtract(activations[-1],groundTruth), self.sigmoid_deriv(z_vals[-1]))  
          
        for l in range(-2, -len(self.format),-1):
            gradients["biases"][l] = np.multiply( np.matmul(np.transpose(self.network["weights"][l+1]),gradients["biases"][l+1]) , self.sigmoid_deriv(z_vals[l]))
        
        for l in range(-1, -len(self.format),-1):   
            gradients["weights"][l] = np.dot(np.meshgrid(activations[l-1],gradients["biases"][l])[1],np.diag(activations[l-1]))

        return gradients["weights"], gradients["biases"]

    def train(self, epochs, batchSize, learningRate):
        for e in range(epochs):
            inputs, groundTruths = loadData(batchSize)
            # Creating placeholder sum arrays

            mean_weight_grad = []
            mean_bias_grad = []

            for i in range(len(self.format)-1):
                weightToAppend = np.zeros((self.format[i+1],self.format[i])) 
                mean_weight_grad.append(weightToAppend)

                biasToAppend = np.zeros(self.format[i+1]) 
                mean_bias_grad.append(biasToAppend)
            
            if e%200 == 0:
                store = []
                for j in range(batchSize):
                    store.append(self.cost(groundTruths[0],self.evaluate(inputs[0])))
                print("Step:", e, "Loss:", np.mean(store))
                
          
            # Calculating average gradients

            for j in range(batchSize):
                cur_weights, cur_biases = self.backprop(inputs[j], groundTruths[j])
                for l in range(len(self.format)-1):
                    mean_weight_grad[l] = np.add(mean_weight_grad[l], cur_weights[l]/batchSize) 
                    mean_bias_grad[l] = np.add(mean_bias_grad[l], cur_biases[l]/ batchSize)
                
            # Apply gradients
            for l in range(len(self.format)-1):
                self.network["weights"][l] = np.subtract(self.network["weights"][l], mean_weight_grad[l]* learningRate) 
                self.network["biases"][l] = np.subtract(self.network["biases"][l], mean_bias_grad[l]* learningRate) 

    def test(self):
        inputs, groundTruths = loadData(30, "iris_test.csv")
        successCount = 0

        for i in range(len(inputs)):
            result = self.evaluate(inputs[i])

            # print(inputs[i], groundTruths[i], result)

            if np.argmax(result) == np.argmax(groundTruths[i]):
                successCount += 1

        return successCount/30
