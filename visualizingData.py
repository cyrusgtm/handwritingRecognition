import trainTestSmallNetwork as ttsn
# import neuralNetwork as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2

# This function is used for saving and viewing images made by hand. It should be used
# when you want to display your handwritten numbers that was written in paper
# into 28*28 pixel image.
def saveImage(path, locationAndName, color):
	# It takes three argument: path, locationName and color. Path take the actual
	# path/location of the image to upload it into the program, locationAndName takes
	# the location of a folder where you want to save the new image and name of that
	# image and finally color takes either Greys or None for grey scaled image or
	# colored image.
	imageArray = cv2.imread(path, 0)
	resizedImage = np.amax(imageArray) - cv2.resize(imageArray, (28,28))
	imageData = (resizedImage/255.0*0.99) + 0.01
	plt.imshow(imageData, cmap = color, interpolation = None)
	plt.imsave(locationAndName, imageData, cmap = color)
	# plt.imsave(locationAndName, imageData, cmap = 'Greys')
	plt.show()
	return 'The image was successfully saved and displayed'

print(saveImage(r'C:/Users/F.R.I.E.N.D.S/Desktop/Directed_Studies/images/originalSeven.jpg', 'images/grey_scaled_7', 'Greys'))



# This function is used viewing images made by hand. It should be used
# when you only want to display without saving your handwritten numbers 
# that was written in paper into 28*28 pixel image.
def viewImage(path):
	imageArray = cv2.imread(path, 0)
	resizedImage = np.amax(imageArray) - cv2.resize(imageArray, (28,28))
	imageData = (resizedImage/255.0*0.99) + 0.01
	plt.imshow(imageData, cmap = 'Greys', interpolation = None)
	plt.show()
	return 'The image was successfully saved and displayed'

# print(viewImage(r'C:/Users/F.R.I.E.N.D.S/Desktop/Directed_Studies/images/originalSix.jpg')))



# This function is used viewing images of handwritten digit. It should be used
# when you only want to display, without saving, your handwritten numbers 
# into 28*28 pixel image. Instead of taking a whole image as an argument like
# last two functions, it takes list of arrays that can be changed into an image.
def viewImage2(array):
	resizedImage = np.amax(array) - cv2.resize(imageArray, (28,28))
	imageData = (resizedImage/255.0*0.99) + 0.01
	plt.imshow(imageData, cmap = 'Greys', interpolation = None)
	plt.show()
	return 'The image was successfully saved and displayed'