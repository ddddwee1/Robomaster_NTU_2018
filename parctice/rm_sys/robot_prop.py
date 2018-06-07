# robot info 
port_writing = False
hp = 2000
h = 2
w = 2
ang = 0

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

#motion planning
dst = [25,40]
dst_ang = 10
is_enemy = False

# robot2 
r2h = 0
r2w = 0
r2rot = 0

def get_current_pos():
	return int(h),int(w),int(ang)

def get_order():
	return v1,v2,fw,lr,rot,shoot

def get_r2():
	return r2h,r2w,r2rot
