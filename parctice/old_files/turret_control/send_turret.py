import serial 

def send_msg(byte_array):
	ser = serial.Serial('COM3',115200)
	ser.write(byte_array)
	ser.close()

def print_hex(inp):
	for i in range(len(inp)):
		print(hex(inp[i]),end='|')
	print()

def send_turret(pitch,yaw):
	data_pack = b'\xab\x55'
	pitch = pitch.to_bytes(2,byteorder='big',signed=True)
	yaw = yaw.to_bytes(2,byteorder='big',signed=True)

	data_pack += yaw + pitch 
	check_sum = yaw[0] & yaw[1] & pitch[0] & pitch[1]
	check_sum = yaw[0] + yaw[1] + pitch[0] + pitch[1]
	check_sum = check_sum%256
	data_pack += check_sum.to_bytes(1,byteorder='big') + b'\xff'
	print_hex(data_pack)
	return data_pack

data = send_turret(100,800)
send_msg(data)