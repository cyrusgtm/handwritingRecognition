import numpy as np
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
		# return np.dot(self.who,inputs)
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




