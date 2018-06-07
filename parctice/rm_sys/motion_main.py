import control_test
import util 
import robot_prop
import time 


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

