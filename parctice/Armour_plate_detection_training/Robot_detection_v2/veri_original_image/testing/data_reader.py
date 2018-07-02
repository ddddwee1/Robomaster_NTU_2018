import tensorflow as tf 
import numpy as np 
import model as M 
import random
import cv2
class data_reader():
	def __init__(self,fname):
		print('Reading data...')
		data = []
		f = open(fname)
		for i in f:
			i = i.strip().split('\t')[0]
			img = cv2.imread(i)
			img = cv2.resize(img,(256,256))
			data.append(img)
		self.data = data

	def get_img(self,index):
		return self.data[index]

	def get_size(self):
		return len(self.data)