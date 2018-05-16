import cv2
import numpy as np
import math
import Functions
import tensorflow as tf	

def get_iou(inp1,inp2):	
	x1,y1,w1,h1 = inp1[0],inp1[1],inp1[2],inp1[3]
	x2,y2,w2,h2 = inp2[0],inp2[1],inp2[2],inp2[3]
	xo = min(abs(x1+w1/2-x2+w2/2), abs(x1-w1/2-x2-w2/2))
	yo = min(abs(y1+h1/2-y2+h2/2), abs(y1-h1/2-y2-h2/2))
	if abs(x1-x2) > (w1+w2)/2 or abs(y1-y2) > (h1+h2)/2:
		return 0
	if abs(float((x1-x2)*2)) < abs(w1-w2):
		xo = min(w1, w2)
	if abs(float((y1-y2)*2)) < abs(h1-h2):
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
	return (iou > 0.7) 

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

def crop_original(img,bias_mtx,conf_mtx,b,c,multip):
	"""
	Input: Original Image, Confidence matrix and Bias matrix
	Output: Array of 5 cropped images(size 32x32)
	Description: Select the top 5 regions with highest confidence and return cropped images of those regions  
	"""
	croppedImages = []
	labels = []
	row,col,_ = b.shape
	c = c.reshape([-1])
	indices = c.argsort()[-3:][::-1]      #we get the indices of flattened array in descending order

	for ind in indices:
		#print (ind)
		i = int(np.floor(ind//(col)))
		j = int(np.floor(ind%(col)))
		#print('a:',a,'\tind:',ind,'\ti:',i,'\tj:',j,'\tconf:',c[i][j],'\txbias',b[i][j][0],'\tybias',b[i][j][1])
		# cropping:
		x = int((b[i][j][0]+j*multip+multip//2))
		y = int((b[i][j][1]+i*multip+multip//2))
		w = int(b[i][j][2]/2)
		h = int(b[i][j][3]/2)
#		print('x:',x,'\ty:',y,'\tw:',w,'\th:',h)
		x_low = x-w
		x_high = x+w
		y_low = y-h
		y_high = y+h

		if x_low<0:
			x_low=0
		elif x_high>959:
			x_high=int(959)
		if y_low<0:
			y_low=0
		elif y_high>539:
			y_high=int(539)

		if x_low >= x_high:
			x_low=1
			x_high=int(959)

		if y_low >= y_high:
			y_low=1
			y_high=int(539)

		#print (x_low,x_high, y_low , y_high)

		cp = img[y_low:y_high,x_low:x_high]
		#a,b,_ = cp.shape
		#print (a,b)
		cp = cv2.resize(cp,(24,24))
		croppedImages.append(cp)

		w = (x_high-x_low)
		h = (y_high-y_low)

		inp1 = [x, y, w, h]
		for r in range(row):
			for c in range(col):
				if conf_mtx[r][c][0] == 1:
					inp2 = bias_mtx[r][c].copy()
					#print("r:", r, "\tc:", c, "\t", inp2)
					inp2[0] = inp2[0]+c*multip+multip//2
					inp2[1] = inp2[1]+r*multip+multip//2
					break
		inp2[2] = inp2[2]
		inp2[3] = inp2[3]
		iou = get_iou(inp1, inp2)
		#print("inp1:", inp1, "\tinp2:", inp2, "\tiou", iou)
		if iou > 0.6:
			label = 1
		else:
			label = 0
		#print (label)
		labels.append(label)
		#cp=np.uint8(cp)
		#img=np.uint8(img)
		#cv2.imshow("cropped", cp)
		#cv2.imshow("img", img)
		#cv2.waitKey(100)

	return croppedImages,labels

def dual_crop_original(img,bias_mtx0,conf_mtx0,bias_mtx2,conf_mtx2,b0,b2,c0,c2):

	croppedImages = []
	labels = []
	

	croppedImages0,labels0 = crop_original(img,bias_mtx0,conf_mtx0,b0,c0,4)
	croppedImages.append(croppedImages0[0])
	labels.append(labels0[0])

	croppedImages2,labels2 = crop_original(img,bias_mtx2,conf_mtx2,b2,c2,1)
	croppedImages.append(croppedImages2[0])
	labels.append(labels2[0])


	return croppedImages,labels

def crop_original_test(img,b,c,multip):
	"""
	Input: Original Image, Confidence matrix and Bias matrix
	Output: Array of 5 cropped images(size 32x32)
	Description: Select the top 5 regions with highest confidence and return cropped images of those regions  
	"""
	croppedImages = []
	labels = []
	inds =[]
	row,col,_ = b.shape
	c = c.reshape([-1])
	indices = c.argsort()[-3:][::-1]      #we get the indices of flattened array in descending order

	for ind in indices:
		#print (ind)
		i = int(np.floor(ind//(col)))
		j = int(np.floor(ind%(col)))
		#print (i,j,col)
		#print('a:',a,'\tind:',ind,'\ti:',i,'\tj:',j,'\tconf:',c[i][j],'\txbias',b[i][j][0],'\tybias',b[i][j][1])
		# cropping:
		x = int((b[i][j][0]+j*multip+multip//2))
		y = int((b[i][j][1]+i*multip+multip//2))
		w = int(b[i][j][2]//2)
		h = int(b[i][j][3]//2)
#		print('x:',x,'\ty:',y,'\tw:',w,'\th:',h)
		x_low = x-w
		x_high = x+w
		y_low = y-h
		y_high = y+h

		if x_low<0:
			x_low=0
		elif x_high>959:
			x_high=int(959)
		if y_low<0:
			y_low=0
		elif y_high>539:
			y_high=int(539)

		if x_low >= x_high:
			x_low=1
			x_high=int(959)

		if y_low >= y_high:
			y_low=1
			y_high=int(539)

		#print (x_low,x_high, y_low , y_high)

		cp = img[y_low:y_high,x_low:x_high]
		#a,b,_ = cp.shape
		#print (a,b)
		cp = cv2.resize(cp,(16,16))
		croppedImages.append(cp)
		inds.append(ind)

	return croppedImages,inds
