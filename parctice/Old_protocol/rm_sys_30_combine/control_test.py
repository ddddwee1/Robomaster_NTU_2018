import serial 
import numpy as np 
import time

ser = serial.Serial('/dev/ttyACM0',115200)

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

def send_order(pitch,yaw,fb,lr,rot,trigger):#changes made on 5.14print
	#print 'order sent',fb,lr,rot
	#print fb,lr
	data = send_turret(pitch,yaw,fb,lr,rot,1,trigger)
	send_msg(data)

def read_uwb(pitch,yaw,fb,lr,rot,trigger):
	request_data = send_turret(pitch,yaw,fb,lr,rot,1,trigger,True) #request the info
	send_msg(request_data)
	while True:
		data = ser.read()
		if data==b'\xab':
			data = ser.read()
			if data==b'\x55':
				data = ser.read(16)
				break
	xy = np.frombuffer(data[:4],dtype=np.int16)
	x,y = xy[0],xy[1]
	yaw = np.frombuffer(data[4:6],dtype=np.uint16)
	yaw = yaw[0]
	#print(data)
	t_pitch = np.frombuffer(data[6:8],dtype=np.int16)[0]
	t_yaw = np.frombuffer(data[8:10],dtype=np.int16)[0]
	process = np.frombuffer(data[10],dtype=np.uint8)[0]
	hp = np.frombuffer(data[11:13],dtype=np.uint16)[0]
	isbuff = np.frombuffer(data[13:15],dtype=np.uint16)[0]
	winner = np.frombuffer(data[15],dtype=np.uint8)[0]
	return x,y,yaw,t_pitch,t_yaw,process,hp,isbuff,winner

def get_output():
	data = ser.read()
	data = format(ord(data),'02x')
	print(data)

if __name__=='__main__':
	for i in range(10):	
		x,y,yaw,t_pitch,t_yaw,process,hp,isbuff,winner = read_uwb(0,50,0,0,0,0)
		print('read data',x,y,yaw,hp,t_pitch,t_yaw)
		time.sleep(0.5)

