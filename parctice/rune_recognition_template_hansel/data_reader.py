import cv2

class data_reader():
	def __init__(self):
		
		self.cap = cv2.VideoCapture(0)
		_,self.img = self.cap.read()

	def read(self):
		return self.img 

