import numpy as np
from load import loadData
from neuralNetwork import nNetwork


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
        l_weights = np.random.rand(3,20) /5 -0.1
        l1_weights = np.random.rand(20,30)/5 -0.1
        l2_weights = np.random.rand(30,4)/5 -0.1

        self.network['weights'] = []
        self.network['weights'].append(l_weights)
        self.network['weights'].append(l1_weights)
        self.network['weights'].append(l2_weights)

        # initiate biases
        l_biases = np.random.rand(3) / 10 -0.05
        l1_biases = np.random.rand(20) / 10 -0.05
        l2_biases = np.random.rand(30) / 10 -0.05

        self.network['biases'] = []
        self.network['biases'].append(l_biases)
        self.network['biases'].append(l1_biases)
        self.network['biases'].append(l2_biases)
    
    def evaluate(self, inpu):
        act = np.copy(inpu)

        for l in range(2, -1, -1):            
            curWeight = self.network["weights"][l]
            curBias = self.network["biases"][l]

            z = np.add( np.matmul(curWeight,act), curBias)
            act = self.sigmoid(z)
            
        return act
    
    def cost(self, groundTruth, output):
        return np.sum(np.square(np.subtract(groundTruth, output))) 

    def backpropold(self, input, groundTruth):
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

                    if(l!=2):
                        dcda_next_comp = dcda * dadz* self.network["weights"][l][n][n2]
                        gradients["activations"][l+1][n2] += dcda_next_comp
                # Change in cost to change in biases
                dzdb = 1
                dcdb = dcda * dadz * dzdb
                gradients["biases"][l][n] = dcdb

            # for n2 in range(len(activations[l+1])):
            #     dzda = self.network["weights"][l][n][n2]
            #     for n in range(len(activations[l])):
            #         dcda = gradients["activations"][l][n]
            #         dadz = self.sigmoid_deriv(z_vals[l][n])
            #         if(l!=2):
            #             gradients["activations"][l+1][n2] += dzda*dcda*dadz

        # print(gradients)
        return gradients["weights"], gradients["biases"]

    def backprop(self, input, groundTruth, mean_weight_grad, mean_bias_grad, batchSize):
        act = np.copy(input)
        activations = [input]
        z_vals = []
        gradients = {"weights": [], "biases": []}        

        # create placeholder for gradient data, calc activation
        for l in range(2, -1, -1):            
            curWeight = self.network["weights"][l]
            curBias = self.network["biases"][l]

            z = np.add( np.matmul(act, np.transpose(curWeight)), curBias)
            act = self.sigmoid(z)
            activations.insert(0, act)
            z_vals.insert(0, z)

            gradients["weights"].insert(0, np.zeros_like(curWeight))
            gradients["biases"].insert(0, np.zeros_like(curBias))

        #Actual Algorithim

        gradients["biases"][0] = 2*np.multiply(np.subtract(activations[0],groundTruth), self.sigmoid_deriv(z_vals[0]))    
        for l in range(2):
            gradients["biases"][l+1] = np.multiply( np.matmul(np.transpose(self.network["weights"][l]),gradients["biases"][l]) , self.sigmoid_deriv(z_vals[l+1]))
        for l in range(3):   
            gradients["weights"][l] = np.dot(np.meshgrid(activations[l+1],gradients["biases"][l])[1],np.diag(activations[l+1]))

        # Modify means
        sem.acquire()

        for l in range(3):
            mean_weight_grad[l] = np.add(mean_weight_grad[l], gradients["weights"][l]/batchSize ) 
            mean_bias_grad[l] = np.add(mean_bias_grad[l], gradients["biases"][l]/ batchSize)

        sem.release()

        return gradients["weights"], gradients["biases"]

    def initLock(self, sem):
        
        sem = semaphore

    def train(self, epochs, batchSize, learningRate):
        with open("./log.txt", 'w') as f:
            for i in range(epochs):
                start_time = time.time()
                inputs, groundTruths = loadData(batchSize)

                # Creating placeholder sum arrays
                l1 = np.zeros((3,20))
                l2 = np.zeros((20,30))
                l3 = np.zeros((30,4))

                mean_weight_grad = [l1, l2, l3]

                l1 = np.zeros(3)
                l2 = np.zeros(20)
                l3 = np.zeros(30)

                mean_bias_grad = [l1, l2, l3]
                
                if i%200 == 0:
                    store = []
                    for j in range(batchSize):
                        store.append(self.cost(groundTruths[0],self.evaluate(inputs[0])))
                    print("Step:", i, "Loss:", np.mean(store))

                # Calculating average gradients

                semaphore = Semaphore()
                pool = Pool(initializer=self.initLock, initargs=(semaphore,))
                for j in range(batchSize):
                    # cur_weights, cur_biases = self.backprop(inputs[j], groundTruths[j])

                    # for l in range(3):
                    #     mean_weight_grad[l] = np.add(mean_weight_grad[l], cur_weights[l]/batchSize ) 
                    #     mean_bias_grad[l] = np.add(mean_bias_grad[l], cur_biases[l]/ batchSize)
                    
                    pool.apply_async(self.backprop, (inputs[j], groundTruths[j], mean_weight_grad, mean_bias_grad, batchSize,))
                
                pool.close()
                pool.join()
                
                time_taken = time.time() - start_time
                f.write(f'time taken to generate step {i}: {time_taken}\n')

                # Apply gradients
                for l in range(3):

            
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
    

def test():
    nt = nNetwork([4,4,3])

    print(nt.test())

    # print(nt.backprop([2.3,5.3,2.3,4.2],[0,0,1]))
    nt.train(10000, 40, 0.05)
    
    print(nt.test())
   

    # gradW,gradB = nt.backprop([6.4,2.8,5.6,2.2], [0, 0, 1])
    # print(gradW,"\n","\n",gradB)
 

if __name__ == "__main__":
    test()