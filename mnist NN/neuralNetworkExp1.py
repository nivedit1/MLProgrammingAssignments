"""mnist Neural Network with 1 Hidden Layer"""
import gzip
import numpy as np
import struct
from matplotlib import pyplot as plt
from datetime import datetime

MNIST_IMAGE_MAGIC = 2051
MNIST_LABEL_MAGIC = 2049
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
LEARNING_RATE = 0.1
ALPHA = 0.9
HIDDEN_NEURON_NUMBER_LIST = [20,50,100]

def read_mnist_images(filename):
    """
    This function converts an mnist image into an input array
    :param filename: The filename to read images
    :return: An image array with individual images along the rows and 784 pixels on the column
    """
    with gzip.open(filename, 'rb') as f:
        magic, number, rows, cols = struct.unpack('>iiii', f.read(16))
        if magic != MNIST_IMAGE_MAGIC:
            raise ValueError("Wrong magic number reading MNIST image file")
        array = np.frombuffer(f.read(), dtype='uint8')
        array = array.reshape((number, rows * cols))
        array = array.astype(float)
        array /= 255.

    return array

def read_mnist_labels(filename):
    """
    Read mnist labels from the ubyte file format
    :param filename: the path to read image labels
    :return: An array with labels as integers for each of the individual images
    """
    with gzip.open(filename, 'rb') as f:
        magic, _ = struct.unpack('>ii', f.read(8))
        if magic != MNIST_LABEL_MAGIC:
            raise ValueError("Wrong magic number reading MNIST label file")
        array = np.frombuffer(f.read(), dtype='uint8')
    array = array.reshape(array.size, 1)

    return array

def sigmoid(value):
    """
    Computes the sigmoid of the passed input

    :param value: The value for which the sigmoid needs to be computed
    :return: Sigmoid of the value
    """
    return (1 / (1 + np.exp(-value)))

def compute_hidden(inputData, hiddenWeights,n):
    """
    Computes the hidden layer neuron values

    :param inputData: The input vector
    :param hiddenWeights: Weights between the input and the hidden layer
    :param n: The number of hidden neurons
    :return: The hidden neuron value
    """

    wDotX = np.dot(hiddenWeights, inputData)
    hiddenInput = sigmoid(wDotX)
    hiddenInput = hiddenInput.reshape(n, 1)

    return hiddenInput

def compute_y(inputData, weights):
    """
    Computes the Y Matrix for a given input vector and weight vector
    :param inputData: Input vector to the perceptron
    :param weights: Weights for the perceptron
    :return:  Computed YMatrix
    """

    wdotX = np.dot(weights, inputData)
    outputData = sigmoid(wdotX)
    predictedData = np.argmax(outputData, axis=0)

    return outputData,predictedData

def weight_update(deltaHidden,deltaOutput,weights,weightsHidden,trainingHidden,inputData,delWeights,delWeightsHidden):
    """
    Performs the weight update according to stochastic gradient descent algorithm

    :param deltaHidden: Del J
    :param deltaOutput: Del K
    :param weights: Weight vector between the hidden layer and the output layer
    :param weightsHidden: Weights between the input layer and the hidden layer
    :param trainingHidden: Computed Hidden neuron values
    :param inputData: Input vector
    :param delWeights: Del Weights to be added to weight vector
    :param delWeightsHidden: Del Hidden weights to be added to weightsHidden vector
    :return: weight vector, weightsHidden vector and delta weights
    """

    #Updating the weight vector between the hidden layer and the output layer
    outputMomentum = ALPHA * delWeights
    hiddenTranspose = np.transpose(trainingHidden)
    deltaOutDotHiddenTranspose = np.dot(deltaOutput,hiddenTranspose)
    delWeights = np.multiply(LEARNING_RATE,deltaOutDotHiddenTranspose)
    delWeights = np.add(delWeights,outputMomentum)
    weights = np.add(weights,delWeights)

    #Updating the weight vector between the input layer and the hidden layer
    hiddenMomentum = ALPHA * delWeightsHidden
    inputTranspose = np.transpose(inputData)
    deltaHiddenDotInputTranspose = np.dot(deltaHidden, inputTranspose)
    delWeightsHidden = np.multiply(LEARNING_RATE, deltaHiddenDotInputTranspose)
    delWeightsHidden = np.add(delWeightsHidden,hiddenMomentum)
    weightsHidden = np.add(weightsHidden, delWeightsHidden)

    return weights,weightsHidden,delWeights,delWeightsHidden

def calculateDelta(trainingHidden,outputData,trainingTarget,weights,n):
    """
    Calculates the delta weights

    :param trainingHidden: Hidden layer of the training data
    :param outputData: The output of the training data
    :param trainingTarget: The target values of the training data
    :param weights: Weight vector between the hidden layer and the output
    :param n: Number of hidden neurons
    :return: Del K and Del J
    """

    #compute Del k
    onesOutput = np.ones((10,1))
    oneMinusOutput = np.subtract(onesOutput,outputData)
    targetMinusOutput = np.subtract(trainingTarget,outputData)
    intermediate = np.multiply(oneMinusOutput,targetMinusOutput)
    deltaOutput = np.multiply(outputData,intermediate)

    #compute Del j
    onesHidden = np.ones(((n + 1), 1))
    oneMinusHidden = np.subtract(onesHidden,trainingHidden)
    hMultipliedOneMinusHidden = np.multiply(trainingHidden,oneMinusHidden)
    weightsTranspose = np.transpose(weights)
    weightDotdeltaOutput = np.dot(weightsTranspose,deltaOutput)
    deltaHidden = np.multiply(hMultipliedOneMinusHidden,weightDotdeltaOutput)
    deltaHidden = np.delete(deltaHidden, (n), axis=0)

    return deltaOutput,deltaHidden

def create_confusion_matrix(targetData, predictedData):
    """
    Creates the confusion Matrix given target and predicted data

    :param targetData: Target Matrix
    :param predictedData: Predicted Matrix
    :return: Updated Confusion Matrix
    """
    confusionMatrix[(int(np.argmax(targetData, axis=0))), int(predictedTestData)] += 1

def create_target_array(label):
    """
    Creates target array for a given label
    :param label: label of the image
    :return: target array
    """
    t = np.full((10,1),0.1)
    t[label,0] = 0.9

    return t

def dual_plot_accuracy(trainingAccuracy, testingAccuracy, n):
    """
    Plots training accuracy vs testing accuracy for a given learning rate

    :param trainingAccuracy: List of training accuracies for an epoch
    :param testingAccuracy: List of testing accuracies for an epoch
    :param n: Number of hidden neurons
    """
    plt.xlabel('Number of Epochs')
    plt.plot(trainingAccuracy, color = "blue", label = 'training accuracy')
    plt.plot(testingAccuracy, color = "green", label = 'testing accuracy')
    plt.title('Training vs Testing Accuracy with ' + str(n) +  ' hidden neurons' )
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1),ncol=2)
    plt.savefig('Exp1_training_vs_testing_neuralNet_with_HiddenNeuron'+str(n)+'.png',bbox_extra_artists=(lgd,), bbox_inches='tight')

if __name__ == '__main__':

    #Setting the experiment start time to current time
    experiment_start_time = datetime.now()

    # Read mnist training and test data and store them in numpy arrays
    trainingInputImgArray = read_mnist_images('samples//' + TRAIN_IMAGES)
    trainingTargetArray = read_mnist_labels('samples//' + TRAIN_LABELS)
    testInputImgArray = read_mnist_images('samples//' + TEST_IMAGES)
    testTargetArray = read_mnist_labels('samples//' + TEST_LABELS)

    # Adding the bias to both training and testing input
    trainingInputImgArray = np.concatenate((trainingInputImgArray, np.ones((60000, 1))), axis=1)
    testInputImgArray = np.concatenate((testInputImgArray, np.ones((10000, 1))), axis=1)

    for HIDDEN_NEURON_NUMBER in HIDDEN_NEURON_NUMBER_LIST:

        #Initializing lists for training and test accuracies
        trainingAccuracyArray = []
        testAccuracyArray = []

        #Initializing weights for both the layers
        weights = np.random.uniform(-0.05, 0.05, size=(10, (HIDDEN_NEURON_NUMBER + 1)))
        weightsHidden = np.random.uniform(-0.05, 0.05, size=(HIDDEN_NEURON_NUMBER, 785))

        #Initializing del weights and del weights hidden for back propagation
        delWeights = np.zeros((10,(HIDDEN_NEURON_NUMBER + 1)))
        delWeightsHidden = np.zeros((HIDDEN_NEURON_NUMBER, 785))

        #Training for 50 epochs
        for epoch in range(0,50):

            #Initialize the confusion matrix for test data
            confusionMatrix = np.zeros((10, 10))
            confusionMatrix = confusionMatrix.astype(int)

            print("Epoch Number: " + str(epoch))

            #Training Data

            #Training Accuracy Accumulator
            trainingHits = 0

            #Iterating through every image in the input array
            for x in range(0, len(trainingInputImgArray)):
                inputArray= trainingInputImgArray[x].reshape((trainingInputImgArray[x].shape[0],1))
                trainingHidden = compute_hidden(inputArray, weightsHidden,HIDDEN_NEURON_NUMBER)
                trainingHidden = np.concatenate((trainingHidden, np.ones((1, 1))), axis=0)
                outputData, predictedData = compute_y(trainingHidden, weights)
                trainingTarget = create_target_array(trainingTargetArray[x])
                if(epoch > 0):
                    deltaOutput, deltaHidden = calculateDelta(trainingHidden,outputData,trainingTarget,weights,HIDDEN_NEURON_NUMBER)
                    weights, weightsHidden, delWeights, delWeightsHidden = weight_update(deltaHidden,deltaOutput,weights,weightsHidden,trainingHidden,inputArray,delWeights, delWeightsHidden)
                if(trainingTarget[predictedData] == 0.9):
                    trainingHits+=1
            trainingAccuracy = (trainingHits/len(trainingInputImgArray))*100
            trainingAccuracyArray.append(trainingAccuracy)
            print("Training Accuracy: ",trainingAccuracy)

            #Test Data

            #Test Accuracy Accumulator
            testHits = 0

            #Iterating through the test images and passing it through the neural network
            for x in range(0, len(testInputImgArray)):
                inputTestArray = testInputImgArray[x].reshape((testInputImgArray[x].shape[0], 1))
                testHidden = compute_hidden(inputTestArray, weightsHidden, HIDDEN_NEURON_NUMBER)
                testHidden = np.concatenate((testHidden, np.ones((1, 1))), axis=0)
                outputTestData, predictedTestData = compute_y(testHidden, weights)

                testTarget = create_target_array(testTargetArray[x])
                create_confusion_matrix(testTarget, predictedTestData)
                if (testTarget[predictedTestData] == 0.9):
                    testHits += 1
            testAccuracy = (testHits / len(testInputImgArray)) * 100
            testAccuracyArray.append(testAccuracy)
            print("Test Accuracy: ", testAccuracy)

        print(confusionMatrix)

        #Plot training vs test accuracy for each experiment
        dual_plot_accuracy(trainingAccuracyArray, testAccuracyArray, HIDDEN_NEURON_NUMBER)
        plt.close()

    print("Total Time taken for the experiment: ",datetime.now()-experiment_start_time)