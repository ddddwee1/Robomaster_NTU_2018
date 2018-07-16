import cv2 
import numpy as np 
import progressbar
import random

class reader():

	def __init__(self,height=540,width=960,scale_range=[0.5,1.0],lower_bound=2,upper_bound=5,index_multiplier=1):
		# set class params
		self.height = height
		self.width = width
		self.scale_range = scale_range
		self.lower_bound = lower_bound
		self.upper_bound = upper_bound
		self.index_multiplier = index_multiplier
		print('Loading images...')
		self.data = []
		# add a progressbar to make it better look
		bar = progressbar.ProgressBar(max_value=5000)
		f = open('annotation.txt')
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
				# print(fname)
				img = cv2.imread(fname)
				if not img is None:
					self.data.append([img,coord])
				else:
					print(fname)
			# update the progressbar
			counter+=1
			bar.update(counter)
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

	def show_img(self,img,coord):
		imgbuff = img.copy()
		for x,y,w,h in coord:
			x = int(x)
			y = int(y)
			w = int(w)
			h = int(h)
			cv2.rectangle(imgbuff,(x,y),(x-w,y-h),(0,0,255),5)
		cv2.imshow('img',imgbuff)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def get_mtx(self,imgsize,coord):
		# lower_bound indicates the log2 of minimum grid size
		# choose the size of each grid
		indices = []
		grid_sizes = []
		coords = []
		for x,y,w,h in coord:
			area = w*h
			index = np.round(np.log2(area)/2.)
			# print(area)
			index = int(np.clip(index, self.lower_bound, self.upper_bound))
			assert isinstance(self.index_multiplier,int)
			# round to 4x gap
			# print(index)
			index = (index - self.lower_bound )//self.index_multiplier
			grid_size = int(np.exp2(self.index_multiplier*index+self.lower_bound))
			# print(index,grid_size)
			# x center and y center
			xc = x-w/2
			yc = y-h/2
			# append into list
			coords.append([xc,yc,w,h])
			grid_sizes.append(grid_size)
			indices.append(index)

		# create dictionary for conf and bias
		# key: indices, value: [conf,bias]
		result_dict = {}
		for i in range(len(indices)):
			height = int(np.ceil(imgsize[0]/grid_sizes[i]))
			width = int(np.ceil(imgsize[1]/grid_sizes[i]))
			# if no key in dictionary, create empty conf and bias array
			if not indices[i] in result_dict:
				bias_empty = np.zeros([height,width,4],np.float32)
				conf_empty = np.zeros([height,width,1],np.float32)
				# print(imgsize,grid_sizes[i])
				result_dict[indices[i]] = [conf_empty,bias_empty]
			# get the column number and row number 
			xc,yc,w,h = coords[i]
			col_num = int(np.floor(xc/float(grid_sizes[i])))
			row_num = int(np.floor(yc/float(grid_sizes[i])))
			# print(height,width,row_num,col_num)
			# comute the bias_x and bias_y
			grid_center_x = col_num*grid_sizes[i]+grid_sizes[i]//2
			grid_center_y = row_num*grid_sizes[i]+grid_sizes[i]//2
			bias_x = xc - grid_center_x
			bias_y = yc - grid_center_y
			# update the bias matrix and conf matrix
			conf_mtx = result_dict[indices[i]][0]
			bias_mtx = result_dict[indices[i]][1]
			conf_mtx[row_num][col_num][0] = 1.
			bias_mtx[row_num][col_num][0] = bias_x
			bias_mtx[row_num][col_num][1] = bias_y
			bias_mtx[row_num][col_num][2] = w
			bias_mtx[row_num][col_num][3] = h
		return result_dict

	def get_img(self):
		# return one single image
		img,coord = random.sample(self.data,1)[0]
		img,coord = self.random_scale(img,coord)
		img,coord = self.random_crop(img,coord)
		result_dict = self.get_mtx(img.shape,coord)
		return img,result_dict

# a = reader()