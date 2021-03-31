import neuralNetwork as nn
import matplotlib.pyplot as plt
import trainTestSmallNetwork as ttsn
import numpy as np

# setting number of input nodes, hidden nodes, and output
# nodes.
input_nodes = 784
hidden_nodes = 100
output_nodes = 10


# setting learning rate
learning_rate = 1
# initializing our main network
network = nn.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# importing training data with 100 rows of data
trainingDataFile = open('data/mnist_train.csv', 'r')
trainingDataList = trainingDataFile.readlines()
trainingDataFile.close

# importing testing data with 10 rows of data
testingDataFile = open('data/mnist_test.csv', 'r')
testingDataList = testingDataFile.readlines()
testingDataFile.close

# learningRate = []
steps = []
efficiency = []
# Number of times you wanna run the program. The more you run it, the more
# the program gets better.
# for epochs in range(10):
# 	# learning_rate += 0.1
# 	epochs += 1
# 	steps.append(epochs)
# 	# network = nn.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
# 	# learningRate.append(learning_rate)
# 	for trainingData in trainingDataList:
# 		splitTrainingData = trainingData.split(',')
# 		inputs = np.asfarray(splitTrainingData[1:])/255.0 * 0.99 + 0.01
# 		targets = np.zeros(output_nodes) + 0.1
# 		targets[int(splitTrainingData[0])] = 0.99
# 		network.train(inputs, targets)


# 	answers = []
# 	# Testing the network. Similar to training the network, first step is
# 	# to iterate through all the rows of testing data and spliting it,
# 	# second step is to normalize the numbers, third step is to use the
# 	# normalized numbers to test our network.
# 	for testingRecord in testingDataList:
# 		splitTestingData = testingRecord.split(',')
# 		inputs = np.asfarray(splitTestingData[1:])/255.0 * 0.99 + 0.01
# 		output = network.query(inputs)
# 		guess = np.argmax(output)		# return the index of the largest number
# 		label = int(splitTestingData[0])
# 		if guess == label:
# 			answers.append(1)
# 		else:
# 			answers.append(0)

# 	efficiency.append((sum(answers)/ len(answers))*100)

# print(efficiency)
# print(learningRate)


## Make and download plots in your computer.
# plt.plot(steps, efficiency)
# xmin, xmax = plt.xlim()
# ymin, ymax = plt.ylim()

# plt.xlim(1, 10)
# plt.ylim(75, 100)
# # # plt.xlabel('Learning Rate')
# plt.xlabel('Epochs')
# plt.ylabel('Efficiency')
# plt.title('Epochs efficiency curve')
# # # plt.savefig('images/learning_rate_efficiency_curve.png')
# plt.savefig('images/epochs_efficiency_curve')
# plt.show()



