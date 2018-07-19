import mp_util.icramap as icramap
from mp_util.control_template import get_order
import robot_prop

def get_next_point(start,dst):
	return icramap.get_next_point(start,dst)

def move_to(dst=None,dst_ang=None,curr_pos=None):
	# get info
	rot_multiplier = 100

	if curr_pos is None and not robot_prop.port_writing:
		w,h,ang = robot_prop.get_whang()

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
		print p0,p1,fw,lr

	# get rot
	if dst_ang==ang:
		rot = 0
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
		rot = rotation_err * rot_multiplier
		
	robot_prop.lr = lr 
	robot_prop.fw = fw
	robot_prop.rot = rot 

	return lr,fw,rot
