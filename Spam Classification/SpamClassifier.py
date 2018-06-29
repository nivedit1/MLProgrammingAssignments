"""Submitted By: Niveditha Venugopal
Collaborated With: Meera Murali
"""

import numpy as np
import math

def convertToNumpyArray(sourceFilePath):
    """
    This function converts the data file into numpy array.

    :param sourceFilePath: The path of the source file
    :return: sourceFileArray which is the source data converted to numpy
    """
    sourceFile = open(sourceFilePath,'r')
    sourceFileArray = np.loadtxt(sourceFile,delimiter=',')
    return sourceFileArray

def createTrainingAndTestData(sourceFileArray):
    """
    This function converts the source array into training and test numpy data and target arrays.

    :param sourceFileArray: Source data in the form of numpy array
    :return: trainingDataArray,trainingTargetArray,testDataArray,testTargetArray
    """
    totalData = len(sourceFileArray)
    trainingTestSize = int(totalData/2)

    #Counting the spam occurrences in the data set
    totalSpamDataSize = ((sourceFileArray[:, -1]) != 0).sum(0)

    #Counting the non spam occurrences in the data set
    totalNonSpamDataSize = totalData - totalSpamDataSize
    testSpamSize = 0
    testNonSpamSize = 0

    #Initializing the test data array
    testDataArray = np.empty((trainingTestSize,58))
    j=0

    #Iterating over the source array and creating the test data array
    for i in range(len(sourceFileArray)):

        #If the test data array reaches the size, break out of the loop
        if i == len(sourceFileArray):
            break

        #Add spam data to test array till half of the total spam data is added to test array
        if sourceFileArray[i][-1] == 1 and testSpamSize < int(totalSpamDataSize/2):
            testSpamSize +=1
            testDataArray[j] = sourceFileArray[i]
            sourceFileArray = np.delete(sourceFileArray,i,axis=0)
            j+=1

        #Add non spam data to test array till half of the total non spam data is added to the test array
        elif sourceFileArray[i][-1] == 0 and testNonSpamSize < int(totalNonSpamDataSize/2):
            testNonSpamSize +=1
            testDataArray[j] = sourceFileArray[i]
            sourceFileArray = np.delete(sourceFileArray, (i), axis=0)
            j+=1

    # Creating the training data array and target array
    trainingDataArray = sourceFileArray
    trainingTargetArray = trainingDataArray[:, -1]
    trainingTargetArray = np.reshape(trainingTargetArray,((len(trainingTargetArray),1)))
    trainingDataArray = trainingDataArray[:, 0:-1]

    #Creating the test data array and target array
    testTargetArray = testDataArray[:,-1]
    testTargetArray = np.reshape(testTargetArray,(len(testTargetArray),1))
    testDataArray = testDataArray[:,0:-1]

    return trainingDataArray,trainingTargetArray,testDataArray,testTargetArray

def computePrior(trainingTargetArray):
    """
    This function computes the prior probabilities of the different classes
    :param trainingTargetArray: Training target array
    :return: Prior probabilities of spam and non spam classes
    """
    spamDataSize = ((trainingTargetArray[:, 0]) != 0).sum(0)
    priorSpam = spamDataSize/len(trainingTargetArray)
    priorNonSpam = (len(trainingTargetArray)-spamDataSize)/len(trainingTargetArray)
    return priorSpam,priorNonSpam

def computeMeanAndSD(trainingDataArray,trainingTargetArray):
    """
    This function computes the class and feature specific mean and standard deviation
    :param trainingDataArray: Training Data Array
    :param trainingTargetArray: Training Target Array
    :return: Mean and Standard Deviation of Spam and Non spam classes
    """

    #Concatenating training and target array
    trainingArray = np.concatenate((trainingDataArray,trainingTargetArray),axis=1)

    #Splitting into training spam and non-spam arrays
    trainingSpamArray = trainingArray[np.where(trainingArray[:,-1]==1)]
    trainingNonSpamArray = trainingArray[np.where(trainingArray[:, -1] == 0)]

    #Computing the mean and standard deviation for both the arrays
    spamMean = trainingSpamArray.mean(axis=0)
    spamMean = spamMean[:-1]
    spamSD = trainingSpamArray.std(axis=0)
    spamSD = spamSD[:-1]
    nonSpamMean = trainingNonSpamArray.mean(axis=0)
    nonSpamMean = nonSpamMean[:-1]
    nonSpamSD = trainingNonSpamArray.std(axis=0)
    nonSpamSD = nonSpamSD[:-1]

    #Replacing 0s with 0.001 to avoid log 0 issue
    spamSD[np.where(spamSD == 0)] = 0.0001
    nonSpamSD[np.where(nonSpamSD == 0)] = 0.0001

    return spamMean,nonSpamMean,spamSD,nonSpamSD

def NBClassifier(mean,SD,testDataArray):
    """
    This function computes the Gaussian Naive Bayes classification
    :param mean: Mean of the class
    :param SD: Standard Deviation of the class
    :param testDataArray: Test Data Array
    :return: Applies the Naiive Bayes Formula and returns the modified test data array
    """
    test_data_length = len(testDataArray)

    #Creating a mean and standard deviation array of the same length as test data
    mean_array = np.array([mean,]*test_data_length)
    SD_array = np.array([SD,]*test_data_length)

    #Applying the Gaussian Naiive Bayes Classification Formula
    NB = np.subtract(testDataArray,mean_array)
    NB = np.square(NB)
    NB = np.multiply(-1,NB)
    SD_array_square = np.square(SD_array)
    two_SD_array_square = np.multiply(2,SD_array_square)
    NB = np.divide(NB,two_SD_array_square)
    NB = np.exp(NB)
    constant_term = 1/math.sqrt(2*math.pi)
    NB = np.multiply(constant_term,NB)
    NB = np.divide(NB,SD_array)
    NB[np.where(NB == 0)] = math.pow(10,-40)
    return NB

def productProbability(NB,prior):
    """
    This function computes the product of the probabilities considering the features are independent
    :param NB: Test data array with Naiive Bayes applied
    :param prior: Prior Probability of the class
    :return: Returns the product of probabilities for class
    """
    #Taking log and summing all the independent probabilities
    NB = np.log(NB)
    cls = np.sum(NB,axis=1)

    #Multiplying the result from the previous step with prior probability of the class
    cls = np.multiply(prior,cls)
    return cls

def predictClass(spamClass,nonSpamClass):
    """
    This function predicts the class of each email example
    :param spamClass: Product of probabilities of spam class
    :param nonSpamClass: Product of probabilities of non-spam class
    :return: Predicted class of test data
    """

    #Applying argmax of spam vs non spam probability for test datum
    predictedClass = np.subtract(spamClass,nonSpamClass)
    predictedClass[np.where(predictedClass<0)] = 0
    predictedClass[np.where(predictedClass > 0)] = 1
    return predictedClass

def computeMetrics(predictedClass,testTargetArray):
    """
    This function computes metrics like accuracy, precision, recall and confusion matrix
    for the test data
    :param predictedClass: Predicted Class of test data
    :param testTargetArray: Actual Class of test data
    :return: Accuracy, Recall, Precision, Confusion Matrix
    """
    predictedClass = np.reshape(predictedClass,(len(predictedClass),1))
    hitsArray = np.equal(predictedClass,testTargetArray)

    #Initializing True Positive, True Negatives, False Positives and False Negatives variables to 0
    TP = TN = FP = FN = 0
    for i in range(len(testTargetArray)):
        if(testTargetArray[i][0] == 1):
            if(hitsArray[i] == True):
                TP += 1
            else:
                FN += 1
        else:
            if(hitsArray[i] == True):
                TN += 1
            else:
                FP += 1

    #Computing accuracy, precision and recall
    precision = (TP/(TP+FP))*100
    recall = (TP/(TP+FN))*100
    accuracy = ((TP+TN)/(TP+TN+FP+FN))*100
    confusionMatrix = np.array([[TP,FN],[FP,TN]])
    return accuracy,precision,recall,confusionMatrix

if __name__=="__main__":

    np.set_printoptions(suppress=True)
    source_file_path = 'data/spambase.data'
    sourceFileArray = convertToNumpyArray(source_file_path)
    trainingDataArray, trainingTargetArray, testDataArray, testTargetArray = createTrainingAndTestData(sourceFileArray)
    priorSpam, priorNonSpam = computePrior(trainingTargetArray)
    spamMean,nonSpamMean,spamSD, nonSpamSD = computeMeanAndSD(trainingDataArray,trainingTargetArray)
    spamNB = NBClassifier(spamMean,spamSD,testDataArray)
    nonSpamNB = NBClassifier(nonSpamMean,nonSpamSD,testDataArray)
    spamClass = productProbability(spamNB,priorSpam)
    nonSpamClass = productProbability(nonSpamNB, priorNonSpam)
    predictedClass = predictClass(spamClass,nonSpamClass)
    accuracy,precision,recall,confusionMatrix = computeMetrics(predictedClass,testTargetArray)
    print("Accuracy: ",accuracy)
    print("Precision: ",precision)
    print("Recall: ", recall)
    print("Confusion Matrix:\n",confusionMatrix)