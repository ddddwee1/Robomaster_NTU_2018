import cv2
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data

WHITE = (255,255,255)

# This index is the index of the image with respect its digit
index = [0,0,0,0,0,0,0,0,0,0]

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

x = mnist.train.images
y = mnist.train.labels

x = x.reshape([-1,28,28])
#print(x.shape)

if not os.path.exists('./DigitImages'):
	os.makedirs('./DigitImages')

for i in range(x.shape[0]):
	buff = x[i]
	# Change the value of mnist image from 0~1 to 0~255
	buff = buff*255
	buff = 255-buff
	# Format the value into 8bit unsigned integer
	buff = np.uint8(buff)
	# Rescale the images' size
	buff_scaled = cv2.resize(buff, (280,160), interpolation = cv2.INTER_CUBIC)
	# Add white border to the left and right of the handwritten digit image so that the image size will be 192 x 120
	buff_bordered = cv2.copyMakeBorder(buff_scaled, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=WHITE)
	# Add the directory/folder if it does not exist
	if not os.path.exists('./DigitImages/Handwritten_%d'%(y[i])):
		os.makedirs('./DigitImages/Handwritten_%d'%(y[i]))
	# Save the image to the respective directory
	cv2.imwrite('./DigitImages/Handwritten_%d/%d_%d.jpg'%(y[i],y[i],index[y[i]]), buff_bordered)
	# Add the index
	index[y[i]] += 1

print(index)

img7Segment = cv2.imread('7-segment-display-numbers.jpg', cv2.IMREAD_COLOR)
imgBordered = cv2.copyMakeBorder(img7Segment, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=WHITE)
imgGray = cv2.cvtColor(imgBordered, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgGray, 240, 255, cv2.THRESH_BINARY_INV)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

if not os.path.exists('./DigitImages/Seven_Segments'):
	os.makedirs('./DigitImages/Seven_Segments')

for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)
	if w*h > 100:
		img = imgBordered[y:y+h, x:x+w]
		imgScaled = cv2.resize(img, (92,124), interpolation = cv2.INTER_LINEAR)
		cv2.imshow('digit',imgScaled)
		k = cv2.waitKey(0) & 0xFF
		k -= 48
		cv2.imwrite('./DigitImages/Seven_Segments/%d.jpg'%(k), imgScaled)
		cv2.destroyAllWindows()

if not os.path.exists('./DigitImages/Flaming_Digits'):
	os.makedirs('./DigitImages/Flaming_Digits')

for i in range(1,6):
	img_ori = cv2.imread('FlamingDigit%d.jpg'%(i), cv2.IMREAD_COLOR)
	imgGray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgGray, 240, 255, cv2.THRESH_BINARY_INV)
	im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	print(hierarchy)
	j = 0
	for cnt in contours:
		if hierarchy[0][j][3] == -1:
			x,y,w,h = cv2.boundingRect(cnt)
			img = img_ori[y:y+h, x:x+w]
			imgScaled = cv2.resize(img, (280,160), interpolation = cv2.INTER_AREA)
			cv2.imshow('digit',imgScaled)
			k = cv2.waitKey(0) & 0xFF
			k -= 48
			cv2.imwrite('./DigitImages/Flaming_Digits/%d.jpg'%(k), imgScaled)
			cv2.destroyAllWindows()
		j += 1
