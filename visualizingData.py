import trainTestSmallNetwork as ttsn
# import neuralNetwork as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2

# trainingSet = tt.trainingDataList
# splitTrainingSet = trainingSet[0].split(',')
# floatTrainingSet = np.asfarray(splitTrainingSet)
# shapeTrainingSet = (floatTrainingSet[1:]/ 255 * 0.99 + 0.01).reshape(28,28)
# plt.imshow(shapeTrainingSet)
# plt.show()
# plt.imsave('images/Colored Number', shapeTrainingSet)
# plt.imsave('images/Grey Scale Number', shapeTrainingSet, cmap = 'Greys')




# Displaying my handwritten number image (5)
path = r'C:/Users/F.R.I.E.N.D.S/Desktop/Directed_Studies/images/originalFive.jpg'
imageArray = cv2.imread(path, 0)
# Displaying my handwritten number image (7)
path = r'C:/Users/F.R.I.E.N.D.S/Desktop/Directed_Studies/images/originalSeven.jpg'
imageArray1 = cv2.imread(path, 0)



# # resizedImage = 255.0 - imageArray.reshape(784)
# # resizing image for 5
# resizedImage = cv2.resize(imageArray, (28,28))

# # resizing image for 7
# resizedImage1 = cv2.resize(imageArray1, (28, 28))


# # print(imageArray.shape)
# # x = 255.0 - resizedImage
# # for 5
# imageData = resizedImage/255.0*0.99 * 0.01

# # for 7
# imageData1 = resizedImage1/255.0*0.99 * 0.01


# # for five
# plt.imshow(imageData, cmap = 'Greys', interpolation = None)
# # plt.show()
# # for seven
# plt.imshow(imageData1, cmap = 'Greys', interpolation = None)
# # plt.show()
# # plt.show()
# plt.imsave('images/My Number Colored 7', imageData1)
# plt.imsave('images/My Number Grey 7', imageData1, cmap = 'Greys')

def saveImage(path, locationAndName):
	imageArray = cv2.imread(path, 0)
	resizedImage = cv2.resize(imageArray, (28,28))
	imageData = (resizedImage/255.0*0.99) + 0.01
	plt.imshow(imageData, cmap = 'Greys', interpolation = None)
	plt.imsave(locationAndName, imageData)
	# plt.imsave(locationAndName, imageData, cmap = 'Greys')
	plt.show()
	return 'The image was successfully saved and displayed'

# saveImage(r'C:/Users/F.R.I.E.N.D.S/Desktop/Directed_Studies/images/originalSeven.jpg', 'images/My Number Colored 7')


def viewImage(path):
	imageArray = cv2.imread(path, 0)
	resizedImage = np.amax(imageArray) - cv2.resize(imageArray, (28,28))
	imageData = (resizedImage/255.0*0.99) + 0.01
	plt.imshow(imageData, cmap = 'Greys', interpolation = None)
	plt.show()
	return 'The image was successfully saved and displayed'

