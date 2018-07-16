#debug lidar function

import serial 
import RPLIDAR_CONST
import numpy as np 
import threading
import time 
from control_template import get_order
from astar import astar
import myslam

fout = open('ffff.txt','w')
	
class data_class():
	def __init__(self):
		
		self.buffer = np.zeros(120)
		self.data = np.zeros(120)
		self.last_ang = 0
		self.last_value = 0
	def record(self,ang,dist):
		#print(ang)
		ind = int(ang)//3
		last_ind = int(self.last_ang)//3
		if ind>119:
			return False
		if self.last_ang>300 and ang<100:
			# self.fill_output(last_ind,360,self.last_value)
			# self.fill_output(0,ind,dist)
			self.data[ind] = dist
			is_new = True
		else:
			# self.fill_output(last_ind,ind,dist)
			self.data[ind] = dist
			is_new = False
		self.last_value = dist
		self.last_ang = ang
		if is_new:
			self.buffer = self.data 
			self.data = np.zeros(120)
		if ind>=120:
			ind = 119
		if ind<0:
			ind = 0
		self.data[ind] = dist
		return is_new
		
	def get_data(self):
		#print('update')
		return list(self.buffer)

	def fill_output(self,start,end,value):
		self.data[start:end] = value

data_saver = data_class()

class myThread(threading.Thread):
	def __init__(self):
		self.data_saver = data_saver
		threading.Thread.__init__(self)

	def run(self):
		print 'starting thread'
		start()
		read_msg()
		stop()
		fout.close()

ser = serial.Serial('/dev/ttyUSB1',115200)
slam = myslam.mySlam(300,4000,120)

def read_msg():
	for _ in range(120000):
	#while 1:
		data = ser.read(5)
		compile_msg(data)

def start():
	start_cmd = RPLIDAR_CONST.HEADER + RPLIDAR_CONST.START
	print(start_cmd)
	ser.write(start_cmd)
	check_start()

def stop():
	stop_cmd = RPLIDAR_CONST.HEADER + RPLIDAR_CONST.STOP
	ser.write(stop_cmd)

def check_start():
	while True:
		data = ser.read()
		print(data)
		if data==b'\xa5':
			data = ser.read()
			if data==b'\x5a':
				data = ser.read(5)
				if data==b'\x05\x00\x00\x40\x81':
					print('-----response-----')
					return True
	# check_start()

def compile_msg(data):
	start_bit = ord(data[1])%2
	#print(start_bit)
	angle = 128 * ord(data[2]) + ord(data[1])//2
	distance = ord(data[4])*128 + ord(data[3])
	angle = angle/64.
	distance = distance/4.
	if_new = data_saver.record(angle,distance)
	if if_new:
		slam.update(data_saver.get_data())
		dt = data_saver.get_data()
		dt = [str(aa) for aa in dt]
		outstr = ' '.join(dt)+'\n'
		fout.write(outstr)
