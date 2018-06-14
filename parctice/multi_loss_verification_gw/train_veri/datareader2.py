import cv2 
import numpy as np 
#import progressbar
import random

class reader():

	def __init__(self,height=540,width=960,scale_range=[0.5,1.0]):
		# set class params
		self.height = height
		self.width = width
		self.scale_range = scale_range
		print('Loading images...')
		self.data = []
		# add a progressbar to make it better look
		#bar = progressbar.ProgressBar(max_value=2579)
		f = open('drive/Colab/multi_loss_verification/train_rpn/annotation.txt')
		counter = 0
		for i in f:
			i = i.strip().split('\t')
			# split the line, get the filename and coordinates 
			fname = i[0]
			coord = i[1:]
			coord = [float(x) for x in coord]
			# split the coordinates 
			x = coord[0::4]
			y = coord[1::4]
			w = coord[2::4]
			h = coord[3::4]
			# combine the coordinates 
			coord = list(zip(x,y,w,h))
			if len(coord)!=0:
				# write into data list
				self.data.append([cv2.imread(fname),coord])
			# update the progressbar
			counter+=1
			print (counter)
		print('Finish reading. Total valid data:',len(self.data))

	def random_crop(self,img,annot):
		# right btm corner
		x2s = [i[0] for i in annot]
		y2s = [i[1] for i in annot]
		# left top corner
		x1s = [i[0]-i[2] for i in annot]
		y1s = [i[1]-i[3] for i in annot]
		# get the shift range
		xmin = np.max(np.array(x2s)) - self.width
		xmax = np.min(np.array(x1s))
		ymin = np.max(np.array(y2s)) - self.height
		ymax = np.min(np.array(y1s))
		# get transform value
		x_trans = random.random()*(xmax-xmin) + xmin
		y_trans = random.random()*(ymax-ymin) + ymin
		# get transformation matrix and do transform
		# print(xmin,xmax)
		M = np.float32([[1,0,-x_trans],[0,1,-y_trans]])
		img_result = img.copy()
		img_result = cv2.warpAffine(img_result,M,(self.width,self.height))
		# substract the transformed pixels
		annot = np.float32(annot) - np.float32([[x_trans,y_trans,0,0]])
		# print(annot)
		return img_result,annot

	def random_scale(self,img,annot):
		# set scale range
		scale_range = self.scale_range
		annot = np.float32(annot)
		scale = random.random()*(scale_range[1]-scale_range[0])+scale_range[0]
		# scaling the annotation and image
		annot = annot * scale
		img_result = cv2.resize(img,None,fx=scale,fy=scale)
		return img_result,annot

	def get_img(self):
		img,coord = random.sample(self.data,1)[0]
		img,coord = self.random_scale(img,coord)
		img,coord = self.random_crop(img,coord)
		return img,coord
