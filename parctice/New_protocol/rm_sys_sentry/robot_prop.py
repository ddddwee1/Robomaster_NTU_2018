# robot info 
port_writing = False

# control 
v1 = 0 #pitch speed
v2 = 0 #yaw speed
shoot = 0
mode = 0

# turret 
t_pitch = 0
t_yaw = 0

# game prop
time_remain = 0

def get_order():
	global v1,v2,mode,shoot,time_remain
#	if time_remain==0:
#		mode = 0
	return v1,v2,mode,shoot
