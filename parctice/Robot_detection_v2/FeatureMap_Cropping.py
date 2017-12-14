import cv2
import numpy as np
import math
import Functions

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

def fmcrop(featureMaps, rpnBias, rpnConf, trueBias):
	"""	
	This is a function that takes the feature map, proposed bias, proposed confidence, and the label to get the cropped feature map, scaled bias, and confidence. So, from the proposed confidence obtained in RPN, the top topAmount will be used. The feature map will be cropped based on the proposed biased. Then the cropped image will be checked if an object is within that cropped image based on the ground truth.
	"""	
	# Initialize the constant information of the input and output
	pickedAmount = 5 # The amount of picked proposed region	
	featureMapsSize = (16,16,256) 
	outputFMsSize = (pickedAmount,3,3,featureMapsSize[2])
	scaledBiasSize = (pickedAmount, 4)

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

