import cv2
import numpy as np
import math
import Functions

def get_iou(inp1,inp2):	
	x1,y1,w1,h1 = inp1[0],inp1[1],inp1[2],inp1[3]
	x2,y2,w2,h2 = inp2[0],inp2[1],inp2[2],inp2[3]
	xo = min(abs(x1+w1/2-x2+w2/2), abs(x1-w1/2-x2-w2/2))
	yo = min(abs(y1+h1/2-y2+h2/2), abs(y1-h1/2-y2-h2/2))
	if abs(x1-x2) > (w1+w2)/2 or abs(y1-y2) > (h1+h2)/2:
		return 0
	if abs(x1-x2) < abs(w1-w2):
		xo = min(w1, w2)
	if abs(y1-y2) < abs(h1-h2):
		yo = min(h1, h2)
	overlap = xo*yo
	total = w1*h1+w2*h2-overlap
	# print(overlap)
	# print(total)
	# print(overlap/total)
	return overlap/total


def crop(inputFM, bias, inputFMSize, outputFMSize):
	"""
	This is a function to crop the region of proposal from the feature map.
	inputFM is in the shape of 16x16x256
	"""	
	# Get the x,y,w,h from the bias
	x,y,w,h = bias[0],bias[1],bias[2],bias[3]
	s = max(w, h)
	tlx = math.floor(x-s/2.0) # Top-Left x-coordinate
	tly = math.floor(y-s/2.0) # Top-Left y-coordinate
	brx = tlx + math.ceil(s/float(inputFMSize[0])) # Bottom-Right x-coordinate + 1
	bry = tly + math.ceil(s/float(inputFMSize[0])) # Bottom-Right y-coordinate + 1
	croppedFM = inputFM[tly:bry, tlx:brx]
	for ctr in range(inputFMSize[2]):
		croppedFM[:,:,ctr] = cv2.resize(croppedFM[:,:,ctr],(outputFMSize[0],outputFMSize[0]))
	scale = float(outputFMSize[0]) / math.ceil(s/inputFMSize[0])
	scaledX = scale*x
	scaledY = scale*y
	scaledW = scale*w
	scaledH = scale*h
	scaledBias = np.array(scaledX, scaledY, scaledW, scaledH)
	return croppedFM, scaledBias

def computeTrueConf(rpnBias, trueBias):
	iou = get_iou(rpnBias, trueBias)
	return (iou > 0.5) 

def biasToCenter(bias):
	"""
	Converting the x,y bias from the coordinate of the bottom right to the coordinate of the center
	"""
	x,y,w,h = bias[0],bias[1],bias[2],bias[3]
	bias[0] = int(x - w/2)
	bias[1] = int(y - h/2) 
	return bias_br 	
	
def fmcrop(featureMaps, rpnBias, rpnConf, trueBias, pickedAmount):
	"""	
	This is a function that takes the feature map, proposed bias, proposed confidence, and the label to get the cropped feature map, scaled bias, and confidence. So, from the proposed confidence obtained in RPN, the top topAmount will be used. The feature map will be cropped based on the proposed biased. Then the cropped image will be checked if an object is within that cropped image based on the ground truth.
	"""	
	# Initialize the constant information of the input and output
	featureMapsSize = (16,16,256) 
	outputFMsSize = (pickedAmount,3,3,featureMapsSize[2])
	scaledBiasSize = (pickedAmount, 4)
	
	# Convert the bias from the bottom right to the center of the bounding rectangle
	rpnBias = biasToCenter(rpnBias)
	trueBias = biasToCenter(trueBias)

	# Picked the top <pickedAmount> based on the rpnConf 
	pickedFlattenIndex = np.argsort(rpnConf,None)[:pickedAmount]

	# Initialize the numpy array tp store the feature maps, scaled bias, and true confidence
	croppedFMs = np.zeros(outputFMsSize)
	scaledBias = np.zeros(scaledBiasSize)
	trueConf = np.zeros(pickedAmount)
	
	# Loop for every picked region
	for ctr in range(pickedAmount):
		# From the flatten index, get the corresponding 2-D Index
		outIdx = math.floor(pickedFlattenIndex[ctr] / featureMapsSize[0])
		innIdx = pickedFlattenIndex[ctr] % featureMapsSize[0]
		croppedFMs[ctr,:,:,:], scaledBias[ctr,:] = crop(featureMaps, rpnBias[outIdx,innIdx], featureMapsSize, outputFMsSize[1:])
		trueConf[ctr] = computeTrueConf(rpnBias[outIdx,innIdx], trueBias)
	return croppedFMs, scaledBias, trueConf

def multipleFeatureMap(featureMaps, rpnBias, rpnConf, trueBias, batchNo):
	"""
	This function serves to take individual image from a batch of images	
	"""
	pickedAmount = 5 # The amount of picked proposed region	
	outputFMs = np.zeros((batchNo*pickedAmount, 3, 3, 256))
	outputConf = np.zeros((batchNo*pickedAmount))
	for ctr in range(batchNo):
		singleFM, _, singleTrueConf = fmcrop(featureMaps[ctr], rpnBias[ctr], rpnConf[ctr], trueBias[ctr], pickedAmount)
		for i in range(pickedAmount):
			outputFMs[ctr*pickedAmount + i] = singleFM[i]
			outputConf[ctr*pickedAmount + i] = outputConf[i]
	return outputFMs, outputConf

def crop_original(img,bias_mtx,conf_mtx,c,b):
	"""
	Input: Original Image, Confidence matrix and Bias matrix
	Output: Array of 5 cropped images(size 32x32)
	Description: Select the top 5 regions with highest confidence and return cropped images of those regions  
	"""
	croppedImages = []
	labels = []
	indices = np.argsort(c,axis=None)      #we get the indices of flattened array in descending order
	for a in range(-1,-6,-1):
		ind = indices[a]
		i = ind//16                 
		j = ind%16
#		print('a:',a,'\tind:',ind,'\ti:',i,'\tj:',j,'\tconf:',c[i][j],'\txbias',int(b[i][j][0]),'\tybias',int(b[i][j][1]))
		# cropping:
		x = int(b[i][j][0])+j*16+8           #x-coord of btm right of a the object center wrt to original image(256x256)
		y = int(b[i][j][1])+i*16+8           #y-coord of btm right of a the object center wrt to original image(256x256)
		w = int(b[i][j][2])                  #half of width of the object
		h = int(b[i][j][3])				     #half of height of the object
		x = x-w                              #x-coord of center of a the object center wrt to original image(256x256)
		y = y-h                              #y-coord of center of a the object center wrt to original image(256x256)
#		print('x:',x,'\ty:',y,'\tw:',w,'\th:',h)
		x_low = x-w
		x_high = x+w
		y_low = y-h
		y_high = y+h
		if x_low<0:
			x_low=0
		elif x_high>255:
			x_high=255
		if y_low<0:
			y_low=0
		elif y_high>255:
			y_high=255
		cp = img[y_low:y_high,x_low:x_high]
		cp = cv2.resize(cp,(32,32))
		croppedImages.append(cp)

		# labelling:

		inp1 = [x, y, 2*w, 2*h]
		for r in range(16):
			for c in range(16):
				if conf_mtx[r][c][0] == 1:
					inp2 = bias_mtx[r][c].copy()
					# print("r:", r, "\tc:", c, "\t", inp2)
					inp2[0] = inp2[0]+c*16+8-inp2[2]
					inp2[1] = inp2[1]+r*16+8-inp2[3]
					break
		inp2[2] = inp2[2]*2
		inp2[3] = inp2[3]*2
		iou = F.get_iou(inp1, inp2)
		# print("inp1:", inp1, "\tinp2:", inp2, "\tiou", iou)
		if iou > 0.5:
			label = 1
		else:
			label = 0
		labels.append(label)

	return croppedImages,labels

def crop_original_test(img,c,b):
	"""
	Input: Original Image, Confidence matrix and Bias matrix
	Output: Array of 5 cropped images(size 32x32)
	Description: Select the top 5 regions with highest confidence and return cropped images of those regions  
	"""
	croppedImages = []
	indices = np.argsort(c,axis=None)      #we get the indices of flattened array in descending order
	inds = []
	for a in range(-1,-6,-1):
		ind = indices[a]
		i = ind//16                 
		j = ind%16
#		print('a:',a,'\tind:',ind,'\ti:',i,'\tj:',j,'\tconf:',c[i][j],'\txbias',int(b[i][j][0]),'\tybias',int(b[i][j][1]))
		# cropping:
		x = int(b[i][j][0])+j*16+8           #x-coord of btm right of a the object center wrt to original image(256x256)
		y = int(b[i][j][1])+i*16+8           #y-coord of btm right of a the object center wrt to original image(256x256)
		w = int(b[i][j][2])                  #half of width of the object
		h = int(b[i][j][3])				     #half of height of the object
		x = x-w                              #x-coord of center of a the object center wrt to original image(256x256)
		y = y-h                              #y-coord of center of a the object center wrt to original image(256x256)
#		print('x:',x,'\ty:',y,'\tw:',w,'\th:',h)
		x_low = x-w
		x_high = x+w
		y_low = y-h
		y_high = y+h
		if x_low<0:
			x_low=0
		elif x_high>255:
			x_high=255
		if y_low<0:
			y_low=0
		elif y_high>255:
			y_high=255
		cp = img[y_low:y_high,x_low:x_high]
		cp = cv2.resize(cp,(32,32))
		croppedImages.append(cp)
		inds.append(ind)

	return croppedImages,inds