import numpy as np
import matplotlib.pyplot as plt
# % matplotlib.inline
import scipy.special

class neuralNetwork:
	# Initialise your neural network
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		self.lr = learningrate
		self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
		
		self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)
		# print(self.who.shape)
		self.activation_function = lambda x:scipy.special.expit(x)
		# self.wih2 = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
		# self.who2 = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
		pass
	# train your neural network
	def train(self, inputs_list, targets_list):
		# Converts inputs list to 2d array
		inputs = np.array(inputs_list, ndmin = 2).T
		targets = np.array(targets_list, ndmin = 2).T

		# calculate signals into hidden layer
		hidden_inputs = np.dot(self.wih, inputs)
		# calculate the signals emerging from hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)

		# calculate signals into final layer
		final_inputs = np.dot(self.who, hidden_outputs)
		# calculate signals emerging from final output layer
		final_outputs = self.activation_function(final_inputs)
		# print(final_outputs)

		# The error between the feed forward output and the actual output
		output_errors = targets - final_outputs
		# print(targets.shape)
		# print(final_outputs.shape)

		# hidden layer error is the output_errors, split by weights recombined at hidden nodes
		hidden_errors = np.dot(self.who.T, output_errors)
		# update the weights for the links between the hidden and output layers
		self.who += self.lr * np.dot((output_errors * final_outputs * (1.0-final_outputs)), np.transpose(hidden_outputs))
		self.wih += self.lr * np.dot((hidden_errors * hidden_outputs *(1.0-hidden_outputs)), np.transpose(inputs))
		# finalResult = np.dot(self.who, inputs)
		return np.sum(output_errors)**2
		# pass
	# query your neural network.  It takes the input to the neural network
	# and return the network's output.
	def query(self, inputs_list):
		# convert inputs to a 2d array
		inputs = np.array(inputs_list, ndmin = 2).T
		# print(inputs)
		# print(inputs.shape)
		# calculate the signal intro hidden layer
		hidden_inputs = np.dot(self.wih, inputs)
		# Calculate the signals emerging from hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)

		# calculate signals into final output layer
		final_inputs = np.dot(self.who, hidden_outputs)
		# calculates the signals emerging from output layer
		final_output = self.activation_function(final_inputs)

		return final_output

		pass

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.2
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
# print(n.query([1.0, 0.5, -1.5]))

# training data with 100 rows
# trainingDataFile = open('mnist_train_100.csv', 'r')
# trainingDataList = trainingDataFile.readlines()
# trainingDataFile.close

# testing data with 10 rows
# testingDataFile = open('mnist_test_10.csv', 'r')
testingDataFile = open('data/mnist_test_10.csv', 'r')
testingDataList = testingDataFile.readlines()
testingDataFile.close()

testingDataList[0] = testingDataList[0][3:]
# trainingDataList[0] = trainingDataList[0][3:]
# testingValues = testingDataList[4].split(',')

# testAsFloats = (np.asfarray(testingValues[1:]) / 255 * 0.99 + 0.01)
# testAsFloats1 = (np.asfarray(testingValues[1:]) / 255 * 0.99 + 0.01).reshape(28,28)
# plt.imshow(testAsFloats1, cmap = 'Greys', interpolation = None)
# plt.show()
# print(testAsFloats)
# print(testingValues)

realTrainingData = open('data/mnist_train_100.csv', 'r')
realTrainingList = realTrainingData.readlines()
realTrainingData.close()

realTrainingList[0] = realTrainingList[0][3:]
# num = 0
# changingAccuracy = []
# changingLearningRate = []
# while num<10:
# 	num += 1

# 	learning_rate += 0.1
# 	changingLearningRate.append(learning_rate)
# 	for realTrainingRecord in realTrainingList:
# 		splitTrainingData = realTrainingRecord.split(',')
# 		inputs = np.asfarray(splitTrainingData[1:])/255 * 0.99 + 0.1
# 		targets = np.zeros(output_nodes) + 0.1
# 		targets[int(splitTrainingData[0])] = 0.99
# 		n.train(inputs, targets)



# # go through all the records in the training data set
# # for trainingRecord in trainingDataList:
# # 	allValues = trainingRecord.split(',')
# # 	inputs = np.asfarray(allValues[1:])/255.0 * 0.99 + 0.01
# # 	targets = np.zeros(output_nodes) + 0.01
# # 	targets[int(allValues[0])] = 0.99
# # 	n.train(inputs, targets)

# 	scorecard = []


# 	for testingRecord in testingDataList:
# 		# print(testingRecord[0])

# 		allTestingValues = testingRecord.split(',')
# 		correctLabel = int(allTestingValues[0])
# 		inputs = np.asfarray(allTestingValues[1:])/ 255.0 * 0.99 + 0.01
# 		outputs = n.query(inputs)
# 		label = np.argmax(outputs)
# 		# print (label)
# 		# print("Network's answer: ", label)
# 		if label == correctLabel:
# 			scorecard.append(1)
# 		else:
# 			scorecard.append(0)
# 	accuracy = sum(scorecard)/len(scorecard)*100
# 	changingAccuracy.append(accuracy)
# 	print('Accuracy rate: ', accuracy)
# # print(scorecard)
# plt.plot(changingLearningRate, changingAccuracy)
# xmin, xmax = plt.xlim()
# ymin, ymax = plt.ylim()

# plt.xlim(0, 1)
# plt.ylim(8, 96)
# plt.show()


	# print()




# Change the numbers into float and reshape the list into 28 * 28 matrix
# imageArray = np.asfarray(allValues[1:]).reshape(28,28)
# plt.imshow(imageArray, cmap = 'Greys', interpolation = 'None')
# plt.show()
# print(imageArray)


# scaling the input
# scaledInput = np.asfarray(allValues[1:])/255.0*0.99 + 0.01

# # setting the final labels. 
# onodes = 10
# targets = np.zeros(onodes) + 0.01
# targets[int(allValues[0])] = 0.99

# print(targets)




for realTrainingRecord in realTrainingList:
	splitTrainingData = realTrainingRecord.split(',')
	inputs = np.asfarray(splitTrainingData[1:])/255 * 0.99 + 0.01
	targets = np.zeros(output_nodes) + 0.1
	targets[int(splitTrainingData[0])] = 0.99
	n.train(inputs, targets)
		



scorecard = []


for testingRecord in testingDataList:
	# print(testingRecord[0])

	allTestingValues = testingRecord.split(',')
	correctLabel = int(allTestingValues[0])
	inputs = np.asfarray(allTestingValues[1:])/ 255.0 * 0.99 + 0.01
	outputs = n.query(inputs)
	label = np.argmax(outputs)
	# print (label)
	# print("Network's answer: ", label)
	if label == correctLabel:
		scorecard.append(1)
	else:
		scorecard.append(0)
accuracy = sum(scorecard)/len(scorecard)*100
# changingAccuracy.append(accuracy)
print('Accuracy rate: ', accuracy)


# import cv2
# imageArray = cv2.imread('images/originalFive.jpg', 0)
# # print(imageArray.shape)

# # resizedImage = 255.0 - imageArray.reshape(784)
# resizedImage = cv2.resize(imageArray, (28,28))

# # print(imageArray.shape)
# # x = 255.0 - resizedImage
# imageData = resizedImage/255.0*0.99 * 0.01
# imageFlatten = imageData.reshape(-1)


# outputs = n.query(imageFlatten)
# label = np.argmax(outputs)
# print(label)
# plt.imshow(imageData, cmap = 'Greys', interpolation = None)
# plt.show()
# # print(imageData)


# # look at the image
# cv2.imshow('image', imageArray)
# # displays image for 1 second(1000ms)
# cv2.waitKey(1000)
# cv2.destroyAllWindows()

# imageData = 255.0 - imageArray.reshape(784)
# imageData = (imageData/255.0 * 0.99)*0.01
# print(imageData)