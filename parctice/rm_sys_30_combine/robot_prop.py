# robot info 
port_writing = False

# control 
v1 = 0 #pitch speed
v2 = 0 #yaw speed
shoot = 0
lr = 0
fw = 0
rot = 0

# turret 
t_pitch = 0
t_yaw = 0

# game prop
process = 0
isbuff = 0
winner = 0
mode = 0

def get_order():
	return v1,v2,fw,lr,rot,shoot

