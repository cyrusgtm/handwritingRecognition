import neuralNetwork as nn
import matplotlib.pyplot as plt
import numpy as np

# setting number of input nodes, hidden nodes, and output
# nodes.
input_nodes = 784
hidden_nodes = 100
output_nodes = 10


# setting learning rate
learning_rate = 0.2
# initializing our main network
network = nn.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)



# Training the network. First step is to loop through the all rows of data,
# second step is to split the data with comma, third step is to normalize 
# the data (convert all the numbers between 0 to 1), fourth step is to
# create our target matrix(output nodes), fifth step is to assign 0.99 to the
# actual number that the row of our data represents and 0.01 to all other numbers,
# and final step is to train our neural network with the input that we got from
# third step and targets.
def training(trainingDataLists):
	for trainingData in trainingDataLists:
		splitTrainingData = trainingData.split(',')
		inputs = np.asfarray(splitTrainingData[1:])/255.0 * 0.99 + 0.01
		targets = np.zeros(output_nodes) + 0.1
		targets[int(splitTrainingData[0])] = 0.99
		network.train(inputs, targets)

# importing training data with 100 rows of data
trainingDataFile = open('data/mnist_train_100.csv', 'r')
trainingDataList = trainingDataFile.readlines()
trainingDataFile.close()
# removing first three element of the first row of our data set.
# the three element contains symbols which is not useful for our
# program
trainingDataList[0] = trainingDataList[0][3:]

training(trainingDataList)



# Testing your network.
def testing(testingDataLists):
	# dictionary that'll contain networks guess and actual answer
	# for comparision
	result = {}
	# list containing 1 and 0. 0 for wrong answers and 1 for right
	# answer
	answers = []
	# Testing the network. Similar to training the network, first step is
	# to iterate through all the rows of testing data and spliting it,
	# second step is to normalize the numbers, third step is to use the
	# normalized numbers to test our network.
	for testingRecord in testingDataLists:
		splitTestingData = testingRecord.split(',')
		inputs = np.asfarray(splitTestingData[1:])/255.0 * 0.99 + 0.01
		output = network.query(inputs)
		guess = np.argmax(output)		# return the index of the largest number
		label = int(splitTestingData[0])
		result["Actual number = " + str(label)] = "Network's number = " + str(guess)
		if guess == label:
			answers.append(1)
		else:
			answers.append(0)
	
	return "Network's efficiancy: " + str((sum(answers)/ len(answers))*100)

# importing testing data with 10 rows of data
testingDataFile = open('data/mnist_test_10.csv', 'r')
testingDataList = testingDataFile.readlines()
testingDataFile.close()
# removing first three element of the first row of our data set.
# The three element contains symbols which is not useful for our
# program
testingDataList[0] = testingDataList[0][3:]
# print(testingDataList[0])

test = testing(testingDataList)
test



