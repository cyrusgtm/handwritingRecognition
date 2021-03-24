import cv2
import neuralNetwork as nn
import numpy as np
import trainTestSmallNetwork as ttsn
import visualizingData as vd
# setting number of input nodes, hidden nodes, and output
# nodes.
input_nodes = 784
hidden_nodes = 100
output_nodes = 10


# setting learning rate
learning_rate = 0.2
# initializing our main network
network = nn.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


# importing training data with 100 rows of data
trainingDataFile = open('data/mnist_train.csv', 'r')
trainingDataList = trainingDataFile.readlines()
trainingDataFile.close


# Train your data by importing the file where we trained our small data.
# ttsn.training(trainingDataList)
for i in range(10):
	for trainingData in trainingDataList:
		splitTrainingData = trainingData.split(',')
		inputs = (np.asfarray(splitTrainingData[1:])/255 * 0.99) + 0.01
		targets = np.zeros(output_nodes) + 0.01
		targets[int(splitTrainingData[0])] = 0.99
		network.train(inputs, targets)

# Displaying my handwritten number image
path = r'C:/Users/F.R.I.E.N.D.S/Desktop/Directed_Studies/images/originalSeven.jpg'
imageArray = cv2.imread(path, 0)
vd.viewImage(path)
# print(imageArray.shape)

resizedImage = cv2.resize(imageArray, (28,28))
reresizedImage = np.amax(resizedImage)-resizedImage
imageData = (reresizedImage/255.0*0.99) + 0.01
print(imageData)
imageFlatten = imageData.reshape(-1)
# print(imageFlatten.shape)

print(np.argmax(network.query(imageFlatten)))
# print(imageArray.shape)
# x = 255.0 - resizedImage
# imageData = resizedImage/255.0*0.99 * 0.01
# print(imageData)