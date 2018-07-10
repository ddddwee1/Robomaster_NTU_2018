import cv2 
import threading
import time 
import os

def hist_equal(img):
	equ = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	equ[:,:,2] = cv2.equalizeHist(equ[:,:,2])
	equ = cv2.cvtColor(equ, cv2.COLOR_HSV2BGR)
	return equ

class camera_thread(threading.Thread):
	def __init__(self):
		self.cap = cv2.VideoCapture(0)
		#self.cap.set(14, 0.01)  #exposure
		self.cap.set(cv2.CAP_PROP_EXPOSURE,-6.)
		self.cap.set(10, 0.01) #brightness
		_,self.img = self.cap.read()
		self.img = hist_equal(self.img)
		threading.Thread.__init__(self)

	def run(self):
		while True:
			time.sleep(0.02)
			try:
				_,self.img = self.cap.read()
				self.img = hist_equal(self.img)
			except:
				for camera_num in range(10):
					try:
						print('camera_down, trying to read camera: ', camera_num)
						self.cap = cv2.VideoCapture(camera_num)
						self.cap.set(cv2.CAP_PROP_EXPOSURE,-6.)
						self.cap.set(10, 0.01) #brightness
						_,self.img = self.cap.read()
						self.img = hist_equal(self.img)
						break
					except:
						self.cap = None
						time.sleep(0.02)
						continue
				if self.cap is None:
					print('----- CANNOT FIND VALID CAMERA -----')
					time.sleep(1)

	def read(self):
		return self.img 
