import tensorflow as tf 
import numpy as np 
import model as M 
import random
#import progressbar
import cv2

width = int(960)
height = int(512) 
xgrid = int(15) # 1920 _ 1080	960_540	480_270	240_135
ygrid = int(8) # 120 _ 68
scalelimit = 0.4

class data_reader():
	def __init__(self,fname):
		print('Reading data...')
		data = []
		f = open(fname)
		counter = 0
		#bar = progressbar.ProgressBar(max_value=1521)
		for i in f:
			if 'nan' in i:
				continue
			#if counter >50:
			#	break
			i = i.strip()
			i = i.split('\t')
			img = cv2.imread(i[0])
			filename = i[0]
			i = i[1:]
			i = [int(k) for k in i]
			print (filename)
			#counter +=1
			#bar.update(counter)
			#cv2.rectangle(img,(i[0]-i[2],i[1]-i[3]),(i[0],i[1]),(255,0,0),2)
			#img = cv2.resize(img,(960,540))
			#cv2.imshow("cropped", img)
			#cv2.waitKey(0)
			data.append([img,[[float(i[0]),float(i[1]),float(i[2]),float(i[3])]]])
			
		self.data = data

	def random_scale(self,img,lmk):
		# set scale range
		lmk = np.float32(lmk).copy()
		scale = random.random()*(1-0.1)+0.5
		# scaling the annotation and image
		lmk = lmk * scale
		img_result = cv2.resize(img,None,fx=scale,fy=scale)
		return img_result,lmk

	## if the 34*60 img does not have the object, assign zero
	## if the 34*60 img has the object, assign 1
	def get_conf_mtx(self,inp,scaling):

		conf_res = np.zeros([int(ygrid*scaling),int(xgrid*scaling),1])
		for i in range(len(inp)):
			x = inp[i][0]
			y = inp[i][1]
			col = int(np.floor(x/64*scaling))
			row = int(np.floor(y/64*scaling))
			conf_res[row][col][0] = 1
			#print (conf_res)
		return conf_res

	## compile x and y into everygrid
	def get_bbox_mtx(self,inp,scaling):

		bbox_res = np.zeros([int(ygrid*scaling),int(xgrid*scaling),4],dtype=np.float32)
		for i in range(len(inp)):
			x = inp[i][0]
			y = inp[i][1]
			w = inp[i][2]
			h = inp[i][3]
			col = int(np.floor(x/64*scaling))
			row = int(np.floor(y/64*scaling))
			#print (x ,y,w,h,col,row)
			bbox_res[row][col][0] = int((x - col*64/scaling - 32/scaling ))
			bbox_res[row][col][1] = int((y - row*64/scaling - 32/scaling ))
			bbox_res[row][col][2] = w
			bbox_res[row][col][3] = h
			#print (x ,y,w,h,col,row,bbox_res[row][col][0],bbox_res[row][col][1])
		return bbox_res


	def random_trans_img(self,img,lmk,scaling):
		img = np.float32(img).copy()
		lmk = np.float32(lmk).copy()
		ngrid = 1

		x = int(lmk[0][0]-lmk[0][2]/2)
		y = int(lmk[0][1]-lmk[0][3]/2)
		w = int(lmk[0][2]/2)
		h = int(lmk[0][3]/2)


		# right btm corner
		x2s = int(lmk[0][0])
		y2s = int(lmk[0][1])
		# left top corner
		x1s = int(lmk[0][0]-lmk[0][2])
		y1s = int(lmk[0][1]-lmk[0][3])
		# get the shift range
		xmin = np.max(np.array(x2s)) - width
		xmax = np.min(np.array(x1s))
		ymin = np.max(np.array(y2s)) - height
		ymax = np.min(np.array(y1s))
		# get transform value
		x_trans = random.random()*(xmax-xmin) + xmin
		y_trans = random.random()*(ymax-ymin) + ymin

		M = np.float32([[1,0,-x_trans],[0,1,-y_trans]])
		img_result = img.copy()
		img_result=np.uint8(img_result)
		img_result = cv2.warpAffine(img_result,M,(width,height))

		lmk = [[float(x),float(y),float(w),float(h)]] - np.float32([[x_trans,y_trans,0,0]])

#		lmkint = [int(lmk[0][0]),int(lmk[0][1]),int(lmk[0][2]),int(lmk[0][3])]
#		print ([float(nx),float(ny),float(nw),float(nh)])
#		cv2.rectangle(img_result,(lmkint[0]-lmkint[2],lmkint[1]-lmkint[3]),(lmkint[0]+lmkint[2],lmkint[1] + lmkint[3]),(255,0,0),2)
#		while ngrid <= 120:
#			cv2.line(img_result, (int(32/scaling * ngrid) , 0), (int(32/scaling* ngrid), 540), (255, 0, 0), 1)
#			cv2.line(img_result, (0 , int(32/scaling * ngrid)), (960, int(32/scaling* ngrid)), (255, 0, 0), 1)
#			ngrid += 1

		#cv2.imshow("cropped", crop_img)
		#cv2.waitKey(0)

		#print (lmkint)

		return img_result,lmk

	def next_train_batch(self,bsize):
		batch = random.sample(self.data,bsize)
		#batch = self.data
		a = []
		nscale=np.random.randint(0,2)
		#nscale = 1
		scaling = 4**float(nscale)
		for i in batch:
			img,lmk = self.random_scale(i[0],i[1])
			img,lmk = self.random_trans_img(img,lmk,scaling)
			bbox_mtx0 = self.get_bbox_mtx(lmk,4)
			conf_mtx0 = self.get_conf_mtx(lmk,4)
			bbox_mtx2 = self.get_bbox_mtx(lmk,1)
			conf_mtx2 = self.get_conf_mtx(lmk,1)
			#img=np.uint8(img)
			#cv2.imshow("cropped", img)
			#cv2.waitKey(0)
			a.append([img, bbox_mtx0 , conf_mtx0,bbox_mtx2,conf_mtx2])
		return a,nscale

#reader= data_reader('annotation.txt')
#while True:
#	train_batch = reader.next_train_batch(1)

