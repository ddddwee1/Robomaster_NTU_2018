import serial 
import RPLIDAR_CONST

ser = serial.Serial('COM3',115200)
def read_msg():
	while 100:
		data = ser.read(5)
		print(data)

def start():
	start_cmd = RPLIDAR_CONST.HEADER + RPLIDAR_CONST.START
	print(start_cmd)
	ser.write(start_cmd)
	data = ser.read(7)
	print(data)
	print('-----response-----')
	return data

def compile_msg(data):
	angle = data[2] + data[1]
	angle = angle >> 1
	distance = data[4] + data[3]

	angle = np.frombuffer(angle,dtype=np.uint16)
	distance = np.frombuffer(distance,dtype=np.uint16)

	print('ang:',angle/64.)
	print('dist:',distance/4.)

# print()
start()
dat = read_msg()
compile_msg(dat)