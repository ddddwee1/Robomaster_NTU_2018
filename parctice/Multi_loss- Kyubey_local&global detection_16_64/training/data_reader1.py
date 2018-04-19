import tensorflow as tf 
import numpy as np 
import model as M 
import random
#import progressbar
import cv2

xcropsize = int(480)
ycropsize = int(270) 
xgrid = int(60) # 1920 _ 1080	960_540
ygrid = int(34) # 120 _ 68

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
			#if counter >300:
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
			data.append([img,[[float(i[0]),float(i[1]),float(i[2]),float(i[3])]]])
			
		self.data = data

	## if the 34*60 img does not have the object, assign zero
	## if the 34*60 img has the object, assign 1
	def get_conf_mtx(self,inp):

		conf_res = np.zeros([ygrid,xgrid,1])
		for i in range(len(inp)):
			x = inp[i][0]
			y = inp[i][1]
			col = int(np.floor(x/16))
			row = int(np.floor(y/16))
			conf_res[row][col][0] = 1
			#print (conf_res)
		return conf_res

	## compile x and y into everygrid
	def get_bbox_mtx(self,inp):

		bbox_res = np.zeros([ygrid,xgrid,4],dtype=np.float32)
		for i in range(len(inp)):
			x = inp[i][0]
			y = inp[i][1]
			w = inp[i][2]
			h = inp[i][3]
			col = int(np.floor(x/16))
			row = int(np.floor(y/16))
			#print (x ,y,w,h,col,row)
			bbox_res[row][col][0] = int(x - col*16 - 8 )
			bbox_res[row][col][1] = int(y - row*16 - 8 )
			bbox_res[row][col][2] = w
			bbox_res[row][col][3] = h
		return bbox_res


	def random_trans_img(self,img,lmk,scale):
		ngrid = 1

		x = int(lmk[0][0] - lmk[0][2]/2)
		y = int(lmk[0][1] - lmk[0][3]/2)
		w = int(lmk[0][2]/2)
		h = int(lmk[0][3]/2)
		xrand=np.random.randint(-280,280)
		yrand=np.random.randint(-170,170)
		sx = x + xrand
		sy = y + yrand

		if sx<=480: 
			xrand=np.random.randint(290,330)
			sx =x + xrand

		if sy<=270:
			yrand=np.random.randint(80,200)
			sy = y + yrand

		crop_img = img[sy-ycropsize:sy+ycropsize, sx-xcropsize:sx+xcropsize]
		cheight, cwidth, _ = crop_img.shape

		#print (cheight, cwidth, sy ,sx)
		nx = int(((float(xcropsize) - float(xrand)) / float(cwidth) * float(xcropsize) * 2) )
		ny = int(((float(ycropsize) - float(yrand)) / float(cheight) * float(ycropsize) * 2) )
		nw = int((float(w) / float(cwidth) * float(xcropsize) * 2) )
		nh = int((float(h) / float(cheight) * float(ycropsize) * 2) )
		rowsize = int(np.ceil(960/scale))
		heightsize = int(np.ceil(540/scale))
		crop_img = cv2.resize(crop_img,(rowsize,heightsize))
		#print ([float(nx),float(ny),float(nw),float(nh)])
		cv2.rectangle(crop_img,(nx-nw,ny-nh),(nx+nw,ny+nh),(255,0,0),2)
		#while ngrid <= 60:
		#	cv2.line(crop_img, (int(16 * ngrid) , 0), (int(16 * ngrid), 540), (255, 0, 0), 1)
		#	cv2.line(crop_img, (0 , int(16 * ngrid)), (960, int(16 * ngrid)), (255, 0, 0), 1)
		#	ngrid += 1
		
		cv2.imshow("cropped", crop_img)
		cv2.waitKey(0)

		lmk = [[float(nx),float(ny),float(nw),float(nh)]]

		return crop_img,lmk

	def next_train_batch(self,bsize):
		#batch = random.sample(self.data,bsize)
		batch = self.data
		a = []
		#nscale= np.random.randint(0,3)
		nscale = 0
		scale = 2**float(nscale)
		for i in batch:
			img,lmk = self.random_trans_img(i[0],i[1],scale)
			bbox_mtx = self.get_bbox_mtx(lmk)
			conf_mtx = self.get_conf_mtx(lmk)
			#cv2.imshow("cropped", img)
			#cv2.waitKey(0)
			a.append([img, bbox_mtx , conf_mtx])
		return a , nscale

#reader= data_reader('annotation.txt')
#while True:
#	train_batch = reader.next_train_batch(1)
#
