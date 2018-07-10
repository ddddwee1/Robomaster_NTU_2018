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

		self.camera_num = 0
		self.cap = cv2.VideoCapture(self.camera_num)
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
				os.system('reboot')
#				print('camera_down, trying to read camera: ', self.camera_num)
#				self.camera_num = 0
#				try:
#					self.camera_num = 0
#					print('camera_down, trying to read camera: ', self.camera_num)
#					self.cap = cv2.VideoCapture(self.camera_num)
#					self.cap.set(cv2.CAP_PROP_EXPOSURE,-6.)
#					self.cap.set(10, 0.01) #brightness
#					_,self.img = self.cap.read()
#					self.img = hist_equal(self.img)
#				except:
#					print('Trying to initialise camera: ', self.camera_num)
#					pass
#				try:
#					if self.camera_num < 10:
#						print('camera_down, trying to read camera: ', self.camera_num)
#						self.cap = cv2.VideoCapture(self.camera_num)
#						self.camera_num += 1
#						self.cap.set(14, 0.01)  #exposure
#						self.cap.set(10, 0.01) #brightness
#						_,self.img = self.cap.read()
#					else:
#						break
#				except:
#					pass

	def read(self):
		return self.img 
