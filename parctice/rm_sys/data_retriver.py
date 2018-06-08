import threading 
import robot_prop 
import time
import control_test
import util 

def get_whang(pitch,yaw,fb,lr,rot,trigger):
	w,h,ang,t_yaw,t_pitch,process,hp,isbuff,winner = control_test.read_uwb(pitch,yaw,fb,lr,rot,trigger)
	w,h,ang = util.normalize_uwb(w,h,ang)
	robot_prop.process = process
	robot_prop.hp = hp
	robot_prop.t_pitch = t_pitch
	robot_prop.t_yaw = t_yaw
	robot_prop.isbuff = isbuff
	robot_prop.winner = winner
	#print 'isbuff',isbuff
	return w,h,ang

class data_reader_thread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		while True:
			time.sleep(0.05)
			if not robot_prop.port_writing:
				pitch,yaw,fb,lr,rot,trigger = robot_prop.get_order()
				w,h,ang = get_whang(pitch,yaw,fb,lr,rot,trigger)
				robot_prop.ang = ang
				#print 'retrieved angle',ang
				if (h>49 or h<1) or (w>79 or w<1) or ang>40:
					continue
				robot_prop.h = h
				robot_prop.w = w
				robot_prop.ang = ang 
				#print 'retriever',h,w,ang
