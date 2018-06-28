import cv2
import threading
import time

class camera_thread(threading.Thread):
	def __init__(self):
		self.cap = cv2.VideoCapture(0)
		_,self.img = self.cap.read()
		threading.Thread.__init__(self)

	def run(self):
		while True:
			time.sleep(0.03)
			_,self.img = self.cap.read()

	def read(self):
		return self.img
