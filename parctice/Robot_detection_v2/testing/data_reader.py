import tensorflow as tf 
import numpy as np 
import model as M 
import random
import cv2

class data_reader():
	def __init__(self,fname):
		a = 1280	# width of the image
		b = 720	# height of the image
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
			data.append(img)
		self.data = data

	def next_img(self,iter):
		return [self.data[iter]]

	def get_iter(self):
		return len(self.data)