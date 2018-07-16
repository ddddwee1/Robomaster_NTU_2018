import tensorflow as tf 
import numpy as np 
import model as M 
import random
import cv2
class data_reader():
	def __init__(self,fname):
		print('Reading data...')
		data = []
		cnt = 0
		f = cv2.VideoCapture(fname)
		while (f.isOpened()):
			cnt += 1
			if cnt%100==0:
				print(cnt)
			ret, frame = f.read()
			if frame is None:
				break
			frame = cv2.resize(frame,(256,256))
			data.append(frame)
		self.data = data

	def next_img(self,iter):
		return [self.data[iter]]

	def get_size(self):
		return len(self.data)
