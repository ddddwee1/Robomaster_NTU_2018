import serial 
import RPLIDAR_CONST
import numpy as np 
from breezyslam.sensors import Laser
from breezyslam.algorithms import RMHC_SLAM
import numpy as np 
from pgm_utils import pgm_save

MAP_SIZE_PIXELS = 800
MAP_SIZE_METERS = 3

class myLaser(Laser):
	def __init__(self):
		Laser.__init__(self,45,5,360,2000)

slam = RMHC_SLAM(myLaser(), MAP_SIZE_PIXELS, MAP_SIZE_METERS)

# for _ in range(10):
# 	data = np.ones(720,np.int32) * 500
# 	data = list(data)
# 	print(data)
# 	slam.update(data,[0,0,0])
# mapbytes = bytearray(MAP_SIZE_PIXELS * MAP_SIZE_PIXELS)
# slam.getmap(mapbytes)
# mapimg = np.reshape(np.frombuffer(mapbytes, dtype=np.uint8), (MAP_SIZE_PIXELS,MAP_SIZE_PIXELS))
# print(mapimg.sum())
# pgm_save('tst.pgm', mapbytes, (MAP_SIZE_PIXELS, MAP_SIZE_PIXELS))

class data_class():
	def __init__(self):
		self.buffer = np.zeros(45)
		self.data = np.zeros(45)
		self.last_ang = 0
		self.last_value = 0
	def record(self,ang,dist):
		# print(ang)
		ind = int(ang)//8
		last_ind = int(self.last_ang)//8
		if self.last_ang>300 and ang<100:
			self.fill_output(last_ind,360,self.last_value)
			self.fill_output(0,ind,dist)
			is_new = True
		else:
			self.fill_output(last_ind,ind,dist)
			is_new = False
		self.last_value = dist
		self.last_ang = ang
		if is_new:
			self.buffer = self.data 
			self.data = np.zeros(45)
		if ind>=45:
			ind = 44
		if ind<0:
			ind = 0
		self.data[ind] = dist
		return is_new
		
	def get_data(self):
		print('update')
		print(self.buffer)
		return list(self.buffer)

	def fill_output(self,start,end,value):
		self.data[start:end] = value

ser = serial.Serial('COM3',115200)
data_saver = data_class()

def read_msg():
	# while 100:
	for _ in range(10000):
		# if _%100==0:
			# print(_)
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
	data = ser.read()
	if data==b'\xa5':
		data = ser.read()
		if data==b'\x5a':
			data = ser.read(5)
			if data==b'\x05\x00\x00\x40\x81':
				print('-----response-----')
				return True
	check_start()

def compile_msg(data):
	# start_bit = data[1]//128
	# print(start_bit)
	angle = 128 * data[2] + data[1]
	distance = data[4]*128 + data[3]
	angle = angle/64.
	distance = distance/4.
	if_new = data_saver.record(angle,distance)
	if if_new:
		# slam.update(data_saver.get_data())
		slam.update(data_saver.get_data(),[0,0,0])
	# print('ang:',angle/64.)
	# print('dist:',distance/4.)

start()
read_msg()
stop()
mapbytes = bytearray(MAP_SIZE_PIXELS * MAP_SIZE_PIXELS)
slam.getmap(mapbytes)
pgm_save('tst.pgm', mapbytes, (MAP_SIZE_PIXELS, MAP_SIZE_PIXELS))