import cv2
import threading
import time

class camera_thread(threading.Thread):
	def __init__(self):
		self.cap = cv2.VideoCapture(0)
		image_width = 1024
		image_height = 768
		self.cap.set(3, image_width)
		self.cap.set(4, image_height)
		self.cap.set(14, 0.0)  #exposure
		self.cap.set(10, 0.05) #brightness
		self.retval,self.img = self.cap.read()
		threading.Thread.__init__(self)

	def run(self):
		while True:
			time.sleep(0.03)
			_,self.img = self.cap.read()

	def read(self):
		return self.retval,self.img
