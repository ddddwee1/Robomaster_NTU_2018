import tensorflow as tf 
import numpy as np 
import model as M 
import random
#import progressbar
import cv2

width = int(960)
height = int(540) 
xgrid = int(30) # 1920 _ 1080	960_540	480_270	240_135
ygrid = int(17) # 120 _ 68
scalelimit = 0.6

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
			counter +=1
			if counter <700:
				continue
			if counter >1000:
				break
			i = i.strip()
			i = i.split('\t')
			img = cv2.imread(i[0])
			filename = i[0]
			i = i[1:]
			i = [int(k) for k in i]
			print (filename)
			#bar.update(counter)
			#cv2.rectangle(img,(i[0]-i[2],i[1]-i[3]),(i[0],i[1]),(255,0,0),2)
			#img = cv2.resize(img,(960,540))
			#cv2.imshow("cropped", img)
			#cv2.waitKey(0)
			data.append(img)
			
		self.data = data

	def get_img(self,index):
		return self.data[index]

	def get_size(self):
		return len(self.data)


