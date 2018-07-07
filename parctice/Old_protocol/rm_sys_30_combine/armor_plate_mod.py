import robot_prop
import time
import util
import detection_mod
import cv2
import sys, select, termios, tty
import math
from turret_module import turret_thread

turret_thread = turret_thread()
turret_thread.start()

# comment manual adjusting after debugging

settings = termios.tcgetattr(sys.stdin)
termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

#Parameters
pitch_bias = 920
yaw_bias = 0
TARGET_MIN_HEIGHT = 10 #Acceptable minimum height of the target before the turret shoots at it
MIN_PITCH_DELTA = 200
MIN_YAW_DELTA = 200

def getKey():
	tty.setraw(sys.stdin.fileno())
	rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
	if rlist:
		key = sys.stdin.read(1)
	else:
		key = ''

	termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
	return key

def manual_shoot(coord):

	global key

	if key == 'q' :
		print 'shoot button pressed'
		turret_thread.shoot_armour()


def auto_shoot(pitch_delta,yaw_delta,coord):
	if len(coord) > 0:
		target_coord = util.get_nearest_target(coord)
		height = target_coord[3]
		if pitch_delta < MIN_PITCH_DELTA and yaw_delta < MIN_YAW_DELTA and height > TARGET_MIN_HEIGHT:
			robot_prop.shoot = 2
		else:
			robot_prop.shoot = 0

def draw_detection(img,coord):
	for x,y,width,height in coord:
		cv2.rectangle(img,(x-width//2,y-height//2),(x+width//2,y+height//2),(0,255,0),2)
	cv2.imshow('img',img)
	cv2.waitKey(1)

def run(camera_thread):
	global pitch_bias,yaw_bias
	key = getKey()
	img = camera_thread.read()
	cv2.imshow('img',img)
	cv2.waitKey(1)
	coord = detection_mod.get_coord_from_detection(img)
	draw_detection(img, coord)
	# change shoot function
	manual_shoot(coord)


	# draw detection for debugging
	t_pitch = robot_prop.t_pitch
	t_yaw = robot_prop.t_yaw
	y_bias = util.pitchbias_to_ypixel(pitch_bias)
	pitch_delta,yaw_delta = util.get_delta(coord,y_bias)

	# auto_shoot(pitch_delta,yaw_delta,coord)

	if key == '2' :
		pitch_bias-=5

	if key == '1' :
		pitch_bias+=5

	if key == '3' :
		yaw_bias-=5

	if key == '4' :
		yaw_bias+=5

	v1 = t_pitch + pitch_delta *1.0 - pitch_bias
	v2 = t_yaw + yaw_delta *1.4 - yaw_bias

	if abs(v1) >= 2000:
		v1 = 2000 * (v1/abs(v1))

	if abs(v2) >= 6000:
		v2 = 6000 * (v2/abs(v2))

	robot_prop.v1 = v1
	robot_prop.v2 = v2
