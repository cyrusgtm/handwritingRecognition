import trainTestSmallNetwork as ttsn


# importing training data with 100 rows of data
trainingDataFile = open('data/mnist_train.csv', 'r')
trainingDataList = trainingDataFile.readlines()
trainingDataFile.close


# Train your data by importing the file where we trained our small data.
ttsn.training(trainingDataList)



# importing testing data with 10 rows of data
testingDataFile = open('data/mnist_test.csv', 'r')
testingDataList = testingDataFile.readlines()
testingDataFile.close


# Test your data by importing the file where we trained our small data.
ttsn.testing(testingDataList)

