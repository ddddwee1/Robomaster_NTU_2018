import data_retriver
import robot_prop 
import time
import util
import detection_mod
from camera_module import camera_thread_0, camera_thread_1, camera_thread_2
import cv2
import sys, select, termios, tty
import math

data_reader = data_retriver.data_reader_thread()
data_reader.start()

camera_thread_0 = camera_thread_0()
camera_thread_0.start()

camera_thread_1 = camera_thread_1()
camera_thread_1.start()

camera_thread_2 = camera_thread_2()
camera_thread_2.start()

counter_detection = 0
counter_shoot = 0

pitch_bias = 920
yaw_bias = 0

turret_cam_detected = False

def getKey():
	tty.setraw(sys.stdin.fileno())
	rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
	if rlist:
		key = sys.stdin.read(1)
	else:
		key = ''

	termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
	return key

settings = termios.tcgetattr(sys.stdin)
termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

def manual_shoot(coord):
	global counter_shoot
	if len(coord) == 0 and counter_shoot >=10:
		robot_prop.shoot = 0
		counter_shoot = 0
	else:
		if key == 'q' :
			print 'shoot button pressed'
			robot_prop.shoot = 2
			counter_shoot +=1

		else: 
			robot_prop.shoot = 0
def timeout():
	pause_shooting = True
	

def auto_shoot(pitch_delta,yaw_delta):
	# TODO: implement auto shooting here
	if is_shooting == False:
		is_shooting = True

	if pitch_delta < 200 and yaw_delta < 200:
		robot.prop.shoot = 2
	else:
		robot.prop.shoot = 0
	pass

def draw_detection(img,coord):
	for x,y,width,height in coord:
		cv2.rectangle(img,(x-width//2,y-height//2),(x+width//2,y+height//2),(0,255,0),2)
	cv2.imshow('img',img)
	cv2.waitKey(1)

def turret_no_detected():
	img_1 = camera_thread_1.read()
	img_2 = camera_thread_2.read()
	coord_1 = detection_mod.get_coord_from_detection(img_1)
	coord_2 = detection_mod.get_coord_from_detection(img_2)
	if len(coord_1) == 0 and len(coord_2) == 0:
		robot_prop.mode = 0
		return 'none'
	elif len(coord_1)!=0:
		robot_prop.mode = 1
		return 'front'
	else:
		robot_prop.mode = 1
		return 'back'

while True:
	if robot_prop.mode == 0:
		base_detect = turret_no_detected()
		if base_detect == 'front':
			robot_prop.v1 = 0
			robot_prop.v2 = 0
		elif base_detect == 'back':
			robot_prop.v1 = 0
			robot_prop.v2 = 18000
		continue

	shooting_timer = Timer(SHOOTING_MAX_DURATION, shooting_timeout)
	pause_timer = Timer(PAUSE_DURATION, shooting_timeout)
	key = getKey()
	img = camera_thread.read()
	coord = detection_mod.get_coord_from_detection(img)

	# change shoot function 
	manual_shoot(coord)
	# auto_shoot(coord)

	# draw detection for debugging
	t_pitch = robot_prop.t_pitch
	t_yaw = robot_prop.t_yaw
	pitch_delta,yaw_delta = util.get_delta(coord)

	if key == '2' :
		pitch_bias-=5

	if key == '1' :
		pitch_bias+=5

	if key == '3' :
		yaw_bias-=5

	if key == '4' :
		yaw_bias+=5

	v1 = t_pitch + pitch_delta *1.0 - pitch_bias
	v2 = t_yaw + yaw_delta *1.0 - yaw_bias

	if abs(v1) >= 2000:
		v1 = 2000 * (v1/abs(v1))

	if abs(v2) >= 6000:
		v2 = 6000 * (v2/abs(v2))

	robot_prop.v1 = v1
	robot_prop.v2 = v2
