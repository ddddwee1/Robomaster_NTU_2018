import numpy as np 
#import matplotlib.pyplot as plt 
import cv2 

class mySlam():
	def __init__(self,pix,dist_max,freq):
		self.pix = pix
		self.map = np.zeros([pix*2,pix*2],dtype=np.uint8)
		self.dist_to_pix = float(dist_max)/float(pix)
		self.freq = freq
		self.data = np.zeros(pix)
		self.dist_max = dist_max

	def update(self,data):
		assert len(data) == self.freq
		data = [0 if _>self.dist_max else _ for _ in data]
		# self.data = data # to be modified 
		self.update_moving_avg(data)
		self.draw()
		# self.show()

	def draw(self):
		self.map = np.zeros([self.pix*2, self.pix*2],dtype=np.uint8)
		for i in range(self.freq):
			dist = self.data[i]
			if dist>self.dist_max and dist==0:
				continue
			#print(self.data)
			x = np.cos(i*np.pi*2/self.freq) * dist / self.dist_to_pix
			y = np.sin(i*np.pi*2/self.freq) * dist / self.dist_to_pix
			x = int(x) + self.pix
			y = int(y) + self.pix
			# self.map[y][x] = 255
			cv2.circle(self.map,(x,y),4,255,-1)

	def show(self):
		cv2.imshow('plot',self.map)
		cv2.waitKey(5)

	def update_moving_avg(self,data,alpha=0.5):
		data_new = []
		for i in range(self.freq):
			old_ = self.data[i]
			new_ = data[i]
			if old_ == 0:
				data_new.append(new_)
			elif new_ == 0:
				data_new.append(0)
			else:
				new = alpha * old_ + (1-alpha) * new_
				data_new.append(new)
		self.data = data_new
