import tensorflow as tf 
import numpy as np 
import model as M 
import random
import cv2

class data_reader():
	def __init__(self,fname):
		a = 240	# length of the image
		b = 180	# width of the image
		print('Reading data...')
		data = []
		f = open(fname)
		for i in f:
			if 'nan' in i:
				continue
			i = i.strip()
			i = i.split('\t')
			img = cv2.imread(i[0])
			print(i[0])
			img = cv2.resize(img,(256,256))
			i = i[1:]
			i = [float(k) for k in i]
			if i[0]>=a:
				i[0] = a-1
			if i[1]>=b:
				i[1] = b-1
			if i[2]>=a:
				i[2] = a-1
			if i[3]>=b:
				i[3] = b-1
			data.append([img,[[float(i[0])*256/a,float(i[1])*256/b,float(i[2])*128/a,float(i[3])*128/b]]])
		self.data = data

	## if the 16*16 img does not have the object, assign zero
	## if the 16*16 img has the object, assign 1
	def get_conf_mtx(self,inp):
		conf_res = np.zeros([16,16,1])
		for i in range(len(inp)):
			x = inp[i][0]
			y = inp[i][1]
			col = int(np.floor(x/16))
			row = int(np.floor(y/16))
			conf_res[row][col][0] = 1
		return conf_res

	## compile x and y into everygrid
	def get_bbox_mtx(self,inp):
		bbox_res = np.zeros([16,16,4],dtype=np.float32)
		for i in range(len(inp)):
			x = inp[i][0]
			y = inp[i][1]
			w = inp[i][2]
			h = inp[i][3]
			# print(x,y)
			col = int(np.floor(x/16))
			row = int(np.floor(y/16))
			bbox_res[row][col][0] = x - col*16.0 - 8.0
			bbox_res[row][col][1] = y - row*16.0 - 8.0
			bbox_res[row][col][2] = w
			bbox_res[row][col][3] = h
		return bbox_res

	def random_trans_img(self,img,lmk):
		while True:
			scale = np.random.rand()
			# scale = 0.85 + scale/4
			# print(scale)
			scale = 1.0
			lmk = np.float32(lmk).copy()
			lmk = lmk * scale
			scale_hw = int(256*scale)
			x_low = int(25.-lmk[0][0])
			x_high = int(235.-lmk[0][0])
			y_low = int(25.-lmk[0][1])
			y_high = int(235.-lmk[0][1])
			if x_high>x_low and y_high>y_low:
				break
		img = cv2.resize(img,(scale_hw,scale_hw))
		shift_x = np.random.randint(x_low,x_high)
		shift_y = np.random.randint(y_low,y_high)
		M = np.float32([[1,0,shift_x],[0,1,shift_y]])	# some transformation
		img = cv2.warpAffine(img,M,(256,256))			# using opencv function
		lmk = np.float32(lmk).copy()
		lmk += np.float32([[shift_x,shift_y,0,0]])
		return img,lmk

	def next_train_batch(self,bsize):
		batch = random.sample(self.data,bsize)
		a = []
		for i in batch:
			img,lmk = self.random_trans_img(i[0],i[1])
			a.append([img,self.get_bbox_mtx(lmk),self.get_conf_mtx(lmk)])
		return a

## Testing
#reader = data_reader()
#batch = reader.next_train_batch(10)
#for i in range(10):
#	bbox_mtx = batch[i][1]
#	conf_mtx = batch[i][2]

#	for r in range(16):
#		for c in range(16):
#			if conf_mtx[r][c] == 1:
#				print("r:{}\tc:{}\t".format(r, c))
#				print(bbox_mtx[r][c])
#	print()


#	for r in range(16):
#		for c in range(16):
#			if conf_mtx[r][c] == 1:
#				print(1, end=" ")
#			else:
#				print(0, end=" ")
#		print()
#	print("\n")

class data_reader_test():
	def __init__(self,fname):
		a = 240	# length of the image
		b = 180	# width of the image
		print('Reading data...')
		data = []
		f = open(fname)
		for i in f:
			if 'nan' in i:
				continue
			i = i.strip()
			i = i.split('\t')
			img = cv2.imread(i[0])
			print(i[0])
			img = cv2.resize(img,(256,256))
			i = i[1:]
			i = [float(k) for k in i]
			if i[0]>=a:
				i[0] = a-1
			if i[1]>=b:
				i[1] = b-1
			if i[2]>=a:
				i[2] = a-1
			if i[3]>=b:
				i[3] = b-1
			data.append([img,[[float(i[0])*256/a,float(i[1])*256/b,float(i[2])*128/a,float(i[3])*128/b]]])
		self.data = data