from mp_util import icramap
from mp_util.control_template import get_order
import control_test
import util 
from PID.PID import PID
import robot_prop
import time 

rotation_PID = PID(1.,0.,0.)

def get_whang():
	w,h,ang = control_test.read_uwb()
	w,h,ang = util.normalize_uwb(w,h,ang)
	return w,h,ang

def get_next_point(start,dst):
	return icramap.get_next_point(start,dst)

def move_to(dst=None,dst_ang=None,curr_pos=None):
	# get info
	rot_multiplier = 100

	if curr_pos is None and not robot_prop.port_writing:
		# test the time here, if
		t1 = time.time()
		w,h,ang = get_whang()
		t2 = time.time()
		print('Read time',t2-t1)
	else:
		h,w,ang = curr_pos

	# if dst_ang is None, we set dst_ang the same as ang to obtain 0 angualr movement
	if dst_ang is None:
		dst_ang = ang

	# get lr and fw
	p0 = [h,w]
	# if dst is None, means we dont want any position movement here
	if dst is None:
		lr, fw = 0, 0
	else:
		p1 = get_next_point(p0,dst)
		lr,fw = get_order(ang,p0,p1) # left/right and forward/backward

	# get rot
	if dst_ang==ang:
		rot = 0
		rotation_PID.eval(0.)
	else:
		if dst_ang>ang:
			if dst_ang - ang < ang + 40 - dst_ang:
				rotation_err = ang - dst_ang
			else:
				rotation_err = ang + 40 - dst_ang
		else:
			if ang - dst_ang < dst_ang + 40 - ang:
				rotation_err = ang - dst_ang
			else:
				rotation_err = ang - 40 - dst_ang
		rot = rotation_PID.eval(rotation_err) * rot_multiplier
		
	robot_prop.lr = lr 
	robot_prop.fw = fw
	robot_prop.rot = rot 

	return lr,fw,rot
