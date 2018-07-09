import cv2 
import threading
import time 

def hist_equal(img):
	equ = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	equ[:,:,2] = cv2.equalizeHist(equ[:,:,2])
	equ = cv2.cvtColor(equ, cv2.COLOR_HSV2BGR)
	return equ

class camera_thread_0(threading.Thread):
	def __init__(self):
		self.cap = cv2.VideoCapture(0)
		self.cap.set(14,0.01) #exp
		self.cap.set(cv2.CAP_PROP_EXPOSURE,-6.)
		self.cap.set(10,0.1) #bright
		_,self.img = self.cap.read()
		self.img = hist_equal(self.img)
		threading.Thread.__init__(self)

	def run(self):
		while True:
			time.sleep(0.02)
			_,self.img = self.cap.read()

	def read(self):
		return self.img 

class camera_thread_1(threading.Thread):
	def __init__(self):
		self.cap = cv2.VideoCapture(1)
		self.cap.set(14,0.01) #exp
		self.cap.set(cv2.CAP_PROP_EXPOSURE,-6.)
		self.cap.set(10,0.1) #bright
		_,self.img = self.cap.read()
		self.img = hist_equal(self.img)
		threading.Thread.__init__(self)

	def run(self):
		while True:
			time.sleep(0.02)
			_,self.img = self.cap.read()
			self.img = hist_equal(self.img)

	def read(self):
		return self.img 

class camera_thread_2(threading.Thread):
	def __init__(self):
		self.cap = cv2.VideoCapture(2)
		self.cap.set(14,0.01) #exp
		self.cap.set(cv2.CAP_PROP_EXPOSURE,-6.)
		self.cap.set(10,0.1) #bright
		_,self.img = self.cap.read()
		self.img = hist_equal(self.img)
		threading.Thread.__init__(self)

	def run(self):
		while True:
			time.sleep(0.02)
			_,self.img = self.cap.read()
			self.img = hist_equal(self.img)

	def read(self):
		return self.img 
