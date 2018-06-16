import threading 
import robot_prop 
import time
import control_test

def get_data(pitch,yaw,mode,trigger):
	t_yaw,t_pitch,mode,time_remain = control_test.read_data(pitch,yaw,mode,trigger)
	robot_prop.t_pitch = t_pitch
	robot_prop.t_yaw = t_yaw
	robot_prop.mode = mode 
	robot_prop.time_remain = time_remain

class data_reader_thread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		while True:
			time.sleep(0.02)
			if not robot_prop.port_writing:
				pitch,yaw,mode,trigger = robot_prop.get_order()
				try:
					get_data(pitch,yaw,mode,trigger)
				except:
					control_test.ser = None