import serial 
import RPLIDAR_CONST

ser = serial.Serial('COM7',115200)
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

# print()
start()
read_msg()