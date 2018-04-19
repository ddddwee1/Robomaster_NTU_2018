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
		cap = cv2.VideoCapture(fname)
		counter = 0
		#bar = progressbar.ProgressBar(max_value=1521)
		while True:
			counter += 1
			if cap.grab():
				flag, frame = cap.retrieve()
			if not flag:
				continue
			if counter <1000:
				continue
			if counter >1500:
				break
			#bar.update(counter)
			#cv2.rectangle(img,(i[0]-i[2],i[1]-i[3]),(i[0],i[1]),(255,0,0),2)
			#img = cv2.resize(img,(960,540))
			#cv2.imshow("cropped", img)
			#cv2.waitKey(0)
			data.append(frame)
			
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
