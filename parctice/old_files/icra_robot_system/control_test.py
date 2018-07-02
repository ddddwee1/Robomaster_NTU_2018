import serial 
import numpy as np 
import time

ser = serial.Serial('/dev/ttyUSB0',115200)

def send_msg(byte_array):
	ser.write(byte_array)

def send_turret(pitch,yaw,fb,lr,rot,flyw,trigger,request=False):
	if request:
		data_pack = b'\xcd\x66'
	else:
		data_pack = b'\xab\x55'

	data = np.int16([pitch,yaw,lr,fb,rot])
	data2 = np.uint8([flyw,trigger])
	buff = data.tobytes()
	buff2 = data2.tobytes()

	data_pack += buff[1] + buff[0] + buff[3] + buff[2] + buff[5] + buff[4] + buff[7] + buff[6] + buff[9] + buff[8] + buff2[0] + buff2[1]
	check_sum = ord(buff[0]) + ord(buff[1]) + ord(buff[2]) + ord(buff[3])+ ord(buff[4])+ ord(buff[5]) + ord(buff[6]) + ord(buff[7]) + ord(buff[8]) + ord(buff[9]) + ord(buff2[0]) + ord(buff2[1])
	check_sum = np.int16([check_sum]).tobytes()[0]
	data_pack += check_sum + b'\xff'
	return data_pack

def send_order(pitch,yaw,fb,lr,rot,trigger):#changes made on 5.14
	data = send_turret(pitch,yaw,fb,lr,rot,1,trigger)
	send_msg(data)

def read_uwb():
	request_data = send_turret(0,0,0,0,0,0,0,True) #request the info
	send_msg(request_data)
	while True:
		data = ser.read()
		if data==b'\xab':
			data = ser.read()
			if data==b'\x55':
				data = ser.read(6)
				break
	xy = np.frombuffer(data[:4],dtype=np.int16)
	x,y = xy[0],xy[1]
	yaw = np.frombuffer(data[4:6],dtype=np.uint16)
	yaw = yaw[0]
	#print(data)
	return x,y,yaw

def get_output():
	data = ser.read()
	data = format(ord(data),'02x')
	print(data)

if __name__=='__main__':

	for i in range(1):	
		x,y,yaw = read_uwb()
		print(x,y,yaw)

	#for i in range(80):
		#get_output()
