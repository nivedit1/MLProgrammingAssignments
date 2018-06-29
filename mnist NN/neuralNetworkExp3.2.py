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
HIDDEN_NEURON_NUMBER= 100

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

def compute_hidden(inputData, hiddenWeights):
    """
    Computes the hidden layer neuron values
    :param inputData: The input vector
    :param hiddenWeights: Weights between the input and the hidden layer
    :param n: The number of hidden neurons
    :return: The hidden neuron value
    """
    wDotX = np.dot(hiddenWeights, inputData)
    hiddenInput = sigmoid(wDotX)
    hiddenInput = hiddenInput.reshape(HIDDEN_NEURON_NUMBER, 1)

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
     :param a: Alpha Value
     :return: weight vector, weightsHidden vector and delta weights
     """
    outputMomentum = ALPHA * delWeights
    hiddenTranspose = np.transpose(trainingHidden)
    deltaOutDotHiddenTranspose = np.dot(deltaOutput,hiddenTranspose)
    delWeights = np.multiply(LEARNING_RATE,deltaOutDotHiddenTranspose)
    delWeights = np.add(delWeights,outputMomentum)
    weights = np.add(weights,delWeights)

    hiddenMomentum = ALPHA * delWeightsHidden
    inputTranspose = np.transpose(inputData)
    deltaHiddenDotInputTranspose = np.dot(deltaHidden, inputTranspose)
    delWeightsHidden = np.multiply(LEARNING_RATE, deltaHiddenDotInputTranspose)
    delWeightsHidden = np.add(delWeightsHidden,hiddenMomentum)
    weightsHidden = np.add(weightsHidden, delWeightsHidden)

    return weights,weightsHidden,delWeights,delWeightsHidden

def calculateDelta(trainingHidden,outputData,trainingTarget,weights):
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
    onesHidden = np.ones(((HIDDEN_NEURON_NUMBER + 1), 1))
    oneMinusHidden = np.subtract(onesHidden,trainingHidden)
    hMultipliedOneMinusHidden = np.multiply(trainingHidden,oneMinusHidden)
    weightsTranspose = np.transpose(weights)
    weightDotdeltaOutput = np.dot(weightsTranspose,deltaOutput)
    deltaHidden = np.multiply(hMultipliedOneMinusHidden,weightDotdeltaOutput)
    deltaHidden = np.delete(deltaHidden, (HIDDEN_NEURON_NUMBER), axis=0)

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
    t[int(label),0] = 0.9

    return t

def dual_plot_accuracy(trainingAccuracy, testingAccuracy, reduce_factor):
    """
    Plots training accuracy vs testing accuracy for a given learning rate

    :param trainingAccuracy: List of training accuracies for an epoch
    :param testingAccuracy: List of testing accuracies for an epoch
    :param lr: Learning Rate
    """
    plt.xlabel('Number of Epochs')
    plt.plot(trainingAccuracy, color = "blue", label = 'training accuracy')
    plt.plot(testingAccuracy, color = "green", label = 'testing accuracy')
    plt.title('Training vs Testing Accuracy when trained with 1/' + str(reduce_factor) +  ' the training data' )
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1),ncol=2)
    plt.savefig('Exp2_training_vs_testing_neuralNet_with_reduceFactor_'+str(reduce_factor)+'.png',bbox_extra_artists=(lgd,), bbox_inches='tight')


def generate_balanced(imageArray, targetArray, reduce_factor):
    """
    This function creates a balanced training set reduced by the reduced_factor
    :param imageArray: Training Input Image Array
    :param targetArray: Training Target Array
    :param reduce_factor: The factor by which the training data needs to be reduced
    :return: Reduced balanced training input and target
    """
    generate_size = int(len(imageArray)/reduce_factor)
    class_size = generate_size/10
    check_size_array = [0,0,0,0,0,0,0,0,0,0]
    check_count = 0
    balanced_image_array = np.empty((generate_size,785))
    balanced_target_array = np.empty((generate_size, 1))
    j = 0
    for i in range(0,len(imageArray)):
        if check_size_array[targetArray[i][0]] <= class_size:
            balanced_image_array[j] = imageArray[i]
            balanced_target_array[j] = targetArray[i]
            check_size_array[targetArray[i][0]] +=1
            j += 1
        for i in check_size_array:
            if i == class_size:
                check_count += 1
        if(check_count == 10):
            break

    return(balanced_image_array,balanced_target_array)


if __name__ == '__main__':

    experiment_start_time_3dot2 = datetime.now()

    # Read mnist training and test data and store them in numpy arrays
    trainingInputImgArray = read_mnist_images('samples//' + TRAIN_IMAGES)
    trainingTargetArray = read_mnist_labels('samples//' + TRAIN_LABELS)
    testInputImgArray = read_mnist_images('samples//' + TEST_IMAGES)
    testTargetArray = read_mnist_labels('samples//' + TEST_LABELS)

    # Adding the bias to both training and testing input
    trainingInputImgArray = np.concatenate((trainingInputImgArray, np.ones((60000, 1))), axis=1)
    testInputImgArray = np.concatenate((testInputImgArray, np.ones((10000, 1))), axis=1)

    # Experiment 3.2 - Choosing a balanced training set after reducing by 1/4

    print("Experiment 3.2 with 1/4 the training set")

    # Creating a reduced balanced training set - 1/4 the size
    newTrainingInputArrayRF4, newTrainingTargetArrayRF4 = generate_balanced(trainingInputImgArray, trainingTargetArray,4)

    # Initializing lists for training and test accuracies
    trainingAccuracyArray = []
    testAccuracyArray = []

    # Initializing weights for both the layers
    weights = np.random.uniform(-0.05, 0.05, size=(10, (HIDDEN_NEURON_NUMBER + 1)))
    weightsHidden = np.random.uniform(-0.05, 0.05, size=(HIDDEN_NEURON_NUMBER, 785))

    # Initializing del weights and del weights hidden for back propagation
    delWeights = np.zeros((10, (HIDDEN_NEURON_NUMBER + 1)))
    delWeightsHidden = np.zeros((HIDDEN_NEURON_NUMBER, 785))

    # Training for 50 epochs
    for epoch in range(0, 50):

        # Initialize the confusion matrix for test data
        confusionMatrix = np.zeros((10, 10))
        confusionMatrix = confusionMatrix.astype(int)

        print("Epoch Number: " + str(epoch))

        # Training Data

        # Training Accuracy Accumulator
        trainingHits = 0

        # Iterating through every image in the input array
        for x in range(0, len(newTrainingInputArrayRF4)):
            inputArray = newTrainingInputArrayRF4[x].reshape((newTrainingInputArrayRF4[x].shape[0], 1))
            trainingHidden = compute_hidden(inputArray, weightsHidden)
            trainingHidden = np.concatenate((trainingHidden, np.ones((1, 1))), axis=0)
            outputData, predictedData = compute_y(trainingHidden, weights)
            # print(newTrainingTargetArrayRF4[x][0])
            trainingTarget1 = create_target_array(newTrainingTargetArrayRF4[x][0])
            if (epoch > 0):
                deltaOutput, deltaHidden = calculateDelta(trainingHidden, outputData, trainingTarget1, weights)
                weights, weightsHidden, delWeights, delWeightsHidden = weight_update(deltaHidden, deltaOutput, weights,
                                                                                     weightsHidden, trainingHidden,
                                                                                     inputArray, delWeights,
                                                                                     delWeightsHidden)
            if (trainingTarget1[predictedData] == 0.9):
                trainingHits += 1
        trainingAccuracy = (trainingHits / len(newTrainingInputArrayRF4)) * 100
        trainingAccuracyArray.append(trainingAccuracy)
        print("Training Accuracy: ", trainingAccuracy)

        # Test Data

        # Test Accuracy Accumulator
        testHits = 0

        # Iterating through the test images and passing it through the neural network
        for x in range(0, len(testInputImgArray)):
            inputTestArray = testInputImgArray[x].reshape((testInputImgArray[x].shape[0], 1))
            testHidden = compute_hidden(inputTestArray, weightsHidden)
            testHidden = np.concatenate((testHidden, np.ones((1, 1))), axis=0)
            outputTestData, predictedTestData = compute_y(testHidden, weights)

            testTarget = create_target_array(testTargetArray[x])
            create_confusion_matrix(testTarget, predictedTestData)
            if (testTarget[predictedTestData] == 0.9):
                testHits += 1
        testAccuracy = (testHits / len(testInputImgArray)) * 100
        testAccuracyArray.append(testAccuracy)
        print("Test Accuracy: ", testAccuracy)

    print("Confusion Matrix for the test data when the training data was reduced by 1/4")
    print(confusionMatrix)

    # Plot training vs test accuracy for each experiment
    dual_plot_accuracy(trainingAccuracyArray, testAccuracyArray, 4)
    plt.close()

    print("Total Time taken for the experiment 3.1: ", datetime.now() - experiment_start_time_3dot2)