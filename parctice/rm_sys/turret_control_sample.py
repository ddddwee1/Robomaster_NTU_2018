import data_retriver
import robot_prop 
import time

data_reader = data_retriver.data_reader_thread()
data_reader.start()

dst_pitch = -20
dst_yaw = 60

while True:
	robot_prop.v1 = dst_pitch * 100
	robot_prop.v2 = dst_yaw * 100
