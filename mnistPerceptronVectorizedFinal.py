"""mnist Perceptron"""
import gzip
import numpy as np
import struct
from matplotlib import pyplot as plt

MNIST_IMAGE_MAGIC = 2051
MNIST_LABEL_MAGIC = 2049

TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

LEARNING_RATE = [0.001,0.01,0.1]


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


def compute_y(inputData, weights):
    """
    Computes the Y Matrix for a given input vector and weight vector
    :param inputData: Input vector to the perceptron
    :param weights: Weights for the perceptron
    :return:  Computed YMatrix
    """

    outputData = np.dot(weights, inputData)
    predictedData = np.argmax(outputData, axis=0)
    return outputData,predictedData


def weight_update(x,weights, y, t, lr):
    """
    Performs the weight update for the perceptron

    :param x: Input Vector
    :param weights: Weight Vector
    :param y: YMatrix
    :param t: Target Array
    :param lr: Learning Rate
    :return: Updated Weight Vector
    """
    x = np.reshape(x,(785,1))
    delta = np.subtract(t,y)
    delta = lr * delta
    xTranspose = np.transpose(x)
    delta = np.dot(delta, xTranspose)
    weightsUpdated = np.add(weights,delta)
    if((weightsUpdated == weights).all()):
         print("Screwed Up")
    return weightsUpdated

def create_confusion_matrix(targetData, predictedData):
    """
    Creates the confusion Matrix given target and predicted data

    :param targetData: Target Matrix
    :param predictedData: Predicted Matrix
    :return: Updated Confusion Matrix
    """
    confusionMatrix[(int(np.argmax(targetData, axis=0))), (int(np.argmax(predictedData, axis=0)))] += 1

def create_target_array(label):
    """
    Creates target array for a given label
    :param label: label of the image
    :return: target array
    """
    t = np.zeros((10,1))
    t[label,0] = 1
    return t

def dual_plot_accuracy(trainingAccuracy, testingAccuracy, lr):
    """
    Plots training accuracy vs testing accuracy for a given learning rate

    :param trainingAccuracy: List of training accuracies for an epoch
    :param testingAccuracy: List of testing accuracies for an epoch
    :param lr: Learning Rate
    """
    plt.xlabel('Number of Epochs')
    plt.plot(trainingAccuracy, color = "blue", label = 'training accuracy')
    plt.plot(testingAccuracy, color = "green", label = 'testing accuracy')
    plt.title('Training vs Testing Accuracy for Learning Rate : ' + str(lr))
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1),ncol=2)
    plt.savefig('training_vs_testing_'+str(lr)+'.png',bbox_extra_artists=(lgd,), bbox_inches='tight')

if __name__ == '__main__':

    #Read mnist training and test data and store them in numpy arrays
    trainingInputImgArray = read_mnist_images('samples//' + TRAIN_IMAGES)
    trainingTargetArray = read_mnist_labels('samples//' + TRAIN_LABELS)
    testInputImgArray = read_mnist_images('samples//' + TEST_IMAGES)
    testTargetArray = read_mnist_labels('samples//' + TEST_LABELS)

    #Adding the bias to both training and testing input
    trainingInputImgArray = np.concatenate((trainingInputImgArray, np.ones((60000, 1))), axis=1)
    testInputImgArray = np.concatenate((testInputImgArray, np.ones((10000, 1))), axis=1)

    # For each learning rate, Assign the weights randomly and train and test the model after each training example
    for lr in LEARNING_RATE:
        print("Learning Rate : " + str(lr))
        trainingAccList = []
        testingAccList = []
        weights = np.random.uniform(-0.05, 0.05, size=(10, 785))

        # Training Stopping Condition : Running for 70 epochs or the improve rate is less than 0.01
        for epoch in range(0,70):
            confusionMatrix = np.zeros((10, 10))
            confusionMatrix = confusionMatrix.astype(int)

            #Training Section

            #Accumulator for computing Accuracy
            trainingHits = 0
            print("Epoch Number : " + str(epoch))
            for x in range(0, len(trainingInputImgArray)):

                #Compute target array for the input vector
                trainingTarget= create_target_array(trainingTargetArray[x])

                #Compute predicted output from the perceptron for the input vector
                trainingOutput, trainingPredicted = compute_y(trainingInputImgArray[x], weights)
                trainingYMatrix = np.zeros((10, 1))
                trainingYMatrix = np.where(trainingOutput > 0, 1, 0)
                trainingYMatrix = trainingYMatrix.reshape(10, 1)

                #Update weights when predicted output != target otherwise use the same weights for the next training example
                if(trainingTarget[trainingPredicted] !=1):
                    if(epoch > 0):
                        weights = weight_update(trainingInputImgArray[x], weights, trainingYMatrix, trainingTarget, lr)
                else:
                    trainingHits +=1

            #Computing training accuracy
            trainingAcc = round((trainingHits / int(len(trainingInputImgArray))) * 100, 2)
            print("Accuracy for training set : " + str(trainingAcc))
            trainingAccList.append(trainingAcc)

            #Testing Section

            #Accumulator for testing accuracy
            testingHits = 0

            # For every epoch of training, test the model with test data
            for x1 in range(0, len(testInputImgArray)):

                # Creating the target array for the test input vector
                testingTarget= create_target_array(testTargetArray[x1])

                # Compute predicted output from the perceptron for the input vector
                testingOutput, testingPredicted = compute_y(testInputImgArray[x1], weights)
                testingYMatrix = np.zeros((10, 1))
                testingYMatrix = np.where(testingOutput > 0, 1, 0)
                testingYMatrix = testingYMatrix.reshape(10, 1)

                # If predicted = target, increment the testingHits accumulator, No weight Update should be performed on test data
                if(testingTarget[testingPredicted] ==1):
                    testingHits +=1

                # Update the confusion matrix for every input vector
                create_confusion_matrix(testingTarget, testingYMatrix)

            # Computing testing accuracy
            testingAcc = round((testingHits / int(len(testInputImgArray))) * 100, 2)
            print("Accuracy for testing set : " + str(testingAcc))
            testingAccList.append(testingAcc)

            #Stopping condition for training
            if (len(trainingAccList) > 1):
                improve_rate = abs(trainingAccList[-1] - trainingAccList[-2])
                if (improve_rate < 0.01 and trainingAccList[-1] > 80):
                    break

        # Plot trainingAcc vs testingAcc for each learning rate
        dual_plot_accuracy(trainingAccList, testingAccList, lr)
        plt.close()

        #Print confusion matrix for the test data after the training has stopped
        print("Confusion Matrix for Learning Rate : " + str(lr))
        print(confusionMatrix)