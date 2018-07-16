port_writing = False
hp = 1000
h = 10
w = 10
ang = 0
# turret
v1 = 0
v2 = 0
shoot = 0
lr = 0
fw = 0
rot = 0

# robot2 
r2h = 0
r2w = 0
r2rot = 0

def get_current_pos():
	return h,w,ang

def get_order():
	return v1,v2,fw,lr,rot,shoot

def get_r2():
	return r2h,r2w,r2rot
