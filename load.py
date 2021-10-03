import random

def loadData(size, filename="iris_training.csv"):
    inputs = []
    groundTruths = []

    try:
        with open(filename, 'r') as trainFile:
            trainData = trainFile.read()
    except:
        print(f'The file {filename} could not be found/opened.')


    trainArray = trainData.strip().split('\n')
    trainArray.pop(0)

    if(size > len(trainArray)):
        size = len(trainArray)
        print("size greater than sample file. size changed to " + str(len(trainArray)))

    for i in range(size):

        randInt = random.randint(0, len(trainArray) - 1)

        stringdata = trainArray[randInt].split(',')
        curInput = [float(data) for data in stringdata]

        groundTruth = curInput.pop(len(curInput) - 1)
        if groundTruth == 0:
            groundTruths.append([1,0,0])
        elif groundTruth == 1:
            groundTruths.append([0,1,0])
        elif groundTruth == 2:
            groundTruths.append([0,0,1])
        else:
            print("invailid result detected")

        inputs.append(curInput)
        trainArray.pop(randInt)

    return inputs, groundTruths