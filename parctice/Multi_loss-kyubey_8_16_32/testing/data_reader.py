import tensorflow as tf 
import numpy as np 
import model as M 
import random
import cv2

cropsize = int(150)
inputsize = int(128)

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
			if counter <500:
				continue
			if counter >800:
				break
			i = i.strip()
			i = i.split('\t')
			img = cv2.imread(i[0])
			img=cv2.resize(img,(960,540))
			filename = i[0]
			i = i[1:]
			i = [int(k) for k in i]
			print (filename)
			#bar.update(counter)
			data.append(img)
			
		self.data = data

	def get_img(self,index):
		return self.data[index]

	def get_size(self):
		return len(self.data)

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
