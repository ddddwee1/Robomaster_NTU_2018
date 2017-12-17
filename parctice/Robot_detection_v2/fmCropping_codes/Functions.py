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

def crop_featureMaps(featureMaps, rpnBias, rpnConf, labelBias, batchSize):
	"""
	This is the function that crop every images/feature maps in a batch where
	there can be more than 1 original image in a batch.
	The shape and size of each input is determined from the rpn network.
	Any changes to the rpn network graph means that the shape of each input in
	this functions need to be changed
	"""
	pickedSize = 5 	# The amount of proposed region that will be
					# picked based on the highest rpnConf

	# Initializing the size of each input:
	featureMapsSize = featureMaps.shape 	# (batchSize, 16 ,16, 256)
	rpnBiasSize = rpnBias.shape 			# (batchSize, 16, 16, 4)
	rpnConfSize = rpnConf.shape				# (batchSize, 16, 16, 1)
	labelBiasSize = (batchSize, 16, 16, 4)	# (batchSize, 16, 16, 4)
	#print(featureMapsSize, rpnBiasSize, rpnConfSize, labelBiasSize)

	# Initializing the size of each output:
	croppedMapsSize = (batchSize*pickedSize, 4, 4, 256)
	CroppedMapsCoorSize = (batchSize*pickedSize, 2, 2)
	croppedMapsConfSize = (batchSize*pickedSize, 1)

	# Initializing the output of this function
	croppedMaps = np.zeros(croppedMapsSize)
	croppedMapsCoor = np.zeros(CroppedMapsCoorSize, dtype=np.uint16)
	croppedMapsConf = np.zeros(croppedMapsConfSize)

	# Start looping for each featureMaps in batchSize
	for batchNo in range(batchSize):

		# Adjusting and fixing the value of the bias
		rpnBias[batchNo] = bias_adjustment(rpnBias[batchNo])
		labelBias[batchNo] = bias_adjustment(labelBias[batchNo])

		# Find the index of the highest value confidence
		pickedIndices = pick_top_conf(rpnConf[batchNo], pickedSize)

		# Iterate through every index that is picked
		for pickedNo in range(pickedSize):
			outputCtr = batchNo*pickedSize + pickedNo
			# print(outputCtr)
			croppedMaps[outputCtr], croppedMapsCoor[outputCtr] = \
				crop_and_resize(featureMaps[batchNo],
								rpnBias[batchNo],
								pickedIndices[pickedNo],
								croppedMapsSize[1:])

			croppedMapsConf[outputCtr,0] = \
				get_veri_label_conf(rpnBias[batchNo],
									labelBias[batchNo],
									pickedIndices[pickedNo])
	return croppedMaps, croppedMapsCoor, croppedMapsConf

def bias_adjustment(biasMatrix):
	"""
	Adjusting the value of the bias in the biasMatrix.
	From x,y,w,h = right, bottom, half rectangle width, half rectangle height
	To x,y,w,h = center, center, rectangle width, rectangle height
	"""
	for biasRow in biasMatrix:
		for bias in biasRow:
			x,y,w,h = bias[0], bias[1], bias[2], bias[3]
			bias[0] = x - w
			bias[1] = y - h
			bias[2] = 2*w
			bias[3] = 2*h
	return biasMatrix

def pick_top_conf(conf, pickedSize):
	"""
	This is the function that will find the index of the highest value of
	confidence from the confidence matrix conf of size [16,16,1]
	The output is of the size [pickedSize, 2]
	"""
	flattenIndices = np.argsort(conf, None)[-pickedSize:]
	indices = np.zeros((pickedSize,2), dtype=np.uint16)
	for ctr in range(pickedSize):
		indices[ctr,0] = math.floor(flattenIndices[ctr] / conf.shape[1])
		indices[ctr,1] = flattenIndices[ctr] % conf.shape[1]
	return indices

def crop_and_resize(featureMaps, bias, index, croppedMapsSize):
	"""
	This is the function to crop and resize the region of proposal from
	the feature map.
	"""
	# Get the x, y, w, h from the bias at the specified index
	bias = bias[index[0], index[1]]
	x, y, w, h = bias[0], bias[1], bias[2], bias[3]

	# Get the center coordinate from the bias
	x = x + index[1]*featureMaps.shape[1] + featureMaps.shape[1]/2
	y = y + index[0]*featureMaps.shape[0] + featureMaps.shape[0]/2

	# Get the sides of the minimum square that bounds the rectangle of the bias
	s = max(w, h)

	# Get the top-left coordinate
	tlx = math.floor((x - s/2) / featureMaps.shape[1])
	tly = math.floor((y - s/2) / featureMaps.shape[0])

	# Get the bottom-right coordinate
	brx = math.floor((x + s/2) / featureMaps.shape[1])
	bry = math.floor((y + s/2) / featureMaps.shape[0])

	# The next two line is the wrong way to calculate the coordinate of the
	# bottom-right coordinate
	# brx = tlx + math.ceil(s / featureMaps.shape[1]) - 1
	# bry = tly + math.ceil(s / featureMaps.shape[0]) - 1


	# Some shifting if one of the coordinates of the cropping square is
	# beyond the bounds
	if(tly < 0):
		dif = 0 - tly
		tly = tly + dif
		bry = bry + dif
	if(tlx < 0):
		dif = 0 - tlx
		tlx = tlx + dif
		brx = brx + dif
	if(bry > featureMaps.shape[0]):
		dif = bry - featureMaps.shape[0]
		bry = bry - dif
		tly = tly - dif
	if(brx > featureMaps.shape[1]):
		dif = brx - featureMaps.shape[1]
		brx = brx - dif
		tlx = tlx - dif

	# Combine the coordinate of top-left and bottom-right to one
	rectCoor = np.array(((tlx, tly), (brx, bry)))

	# Crop the feature map using numpy array manipulation
	croppedFM = featureMaps[tly:bry+1, tlx:brx+1, :]

	# Create a new container to contain the resized image
	resizedFM = np.zeros(croppedMapsSize)

	# Resize every cropped feature map to 3x3
	for ctr in range(featureMaps.shape[2]):
		# print("1. ", croppedFM.shape)
		# print("2. ", croppedFM[:,:,ctr])
		# print("3. ", resizedFM.shape)
		# print("4. ", rectCoor)
		resizedFM[:,:,ctr] = cv2.resize(croppedFM[:,:,ctr], \
										croppedMapsSize[:2])
	# print("4. Test")
	return resizedFM, rectCoor

def get_veri_label_conf(rpnBias, labelBias, index):
	"""
	Get the confidence value to be used as the label in the verification
	network. The value is based on the iou value between rpnBias and labelBias.
	As the input rpnBias and labelBias is a bias matrix, the value is specified
	based on the index.
	"""
	rpnBias = rpnBias[index[0], index[1]]
	labelBias = labelBias[index[0], index[1]]
	iou = get_iou(rpnBias, labelBias)
	return (iou > 0.3)
