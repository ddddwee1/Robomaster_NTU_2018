import data_retriver
import robot_prop 
import time

data_reader = data_retriver.data_reader_thread()
data_reader.start()

dst_pitch = 10.54
dst_yaw = 45 

while True:
	robot_prop.v1 = dst_pitch * 100
	robot_prop.v2 = dst_yaw * 100