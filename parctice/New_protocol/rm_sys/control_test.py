import serial 
import numpy as np 
import time

ser = None

def check_socket():
	global ser
	if ser is None:
		try:
			ser = serial.Serial('/dev/ttyACM0',115200)
			return True
		except:
			ser = None
			return False
	return True

def send_msg(byte_array):
	ser.write(byte_array)

def send_turret(pitch,yaw,mode,trigger,request=False):
	if request:
		data_pack = b'\xcd\x66'
	else:
		data_pack = b'\xab\x55'

	data = np.int16([pitch,yaw])
	data2 = np.uint8([mode,trigger])
	buff = data.tobytes()
	buff2 = data2.tobytes()

	data_pack += buff[1] + buff[0] + buff[3] + buff[2] + buff2[0] + buff2[1]
	check_sum = ord(buff[0]) + ord(buff[1]) + ord(buff[2]) + ord(buff[3]) + ord(buff2[0]) + ord(buff2[1])
	check_sum = np.int16([check_sum]).tobytes()[0]
	data_pack += check_sum + b'\xff'
	return data_pack

def send_order(pitch,yaw,mode,trigger):#changes made on 5.14print
	data = send_turret(pitch,yaw,mode,trigger)
	send_msg(data)

def read_data(pitch,yaw,mode,trigger):
	is_socket = check_socket()
	if not is_socket:
		return 0,0,0,0

	request_data = send_turret(pitch,yaw,mode,trigger,True) #request the info
	send_msg(request_data)
	while True:
		data = ser.read()
		if data==b'\xab':
			data = ser.read()
			if data==b'\x55':
				data = ser.read(7)
				break

	t_pitch = np.frombuffer(data[0:2],dtype=np.int16)[0]
	t_yaw = np.frombuffer(data[2:4],dtype=np.int16)[0]
	mode = np.frombuffer(data[4],dtype=np.uint8)[0]
	time_remain = np.frombuffer(data[5:7],dtype=np.uint16)[0]
	return t_pitch,t_yaw,mode,time_remain
