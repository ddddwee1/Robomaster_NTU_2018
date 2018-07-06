import cv2 
import threading
import time 

class camera_thread_0(threading.Thread):
	def __init__(self):
		self.cap = cv2.VideoCapture(0)
		self.cap.set(14,0.01) #exp
		self.cap.set(10,0.1) #bright
		_,self.img = self.cap.read()
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
		self.cap.set(10,0.1) #bright
		_,self.img = self.cap.read()
		threading.Thread.__init__(self)

	def run(self):
		while True:
			time.sleep(0.02)
			_,self.img = self.cap.read()

	def read(self):
		return self.img 

class camera_thread_2(threading.Thread):
	def __init__(self):
		self.cap = cv2.VideoCapture(2)
		self.cap.set(14,0.01) #exp
		self.cap.set(10,0.1) #bright
		_,self.img = self.cap.read()
		threading.Thread.__init__(self)

	def run(self):
		while True:
			time.sleep(0.02)
			_,self.img = self.cap.read()

	def read(self):
		return self.img 
