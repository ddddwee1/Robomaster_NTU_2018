import threading 
import robot_prop 
import time
import motion_main

class data_reader_thread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		while True:
			time.sleep(0.05)
			if not robot_prop.port_writing:
				pitch,yaw,fb,lr,rot,trigger = robot_prop.get_order()
				w,h,ang = motion_main.get_whang(pitch,yaw,fb,lr,rot,trigger)
				robot_prop.h = h
				robot_prop.w = w
				robot_prop.ang = ang 
				#print h,w,ang
