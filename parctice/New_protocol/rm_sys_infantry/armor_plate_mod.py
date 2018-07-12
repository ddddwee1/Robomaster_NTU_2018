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
pitch_bias = 250
yaw_bias = 80
TARGET_MIN_HEIGHT = 12 #Acceptable minimum height of the target before the turret shoots at it
MIN_PITCH_DELTA = 200
MIN_YAW_DELTA = 100
MIN_CONSECUTIVE_TARGET_LOCKS = 2
pitch_weight = 1.3
yaw_weight = 2.0

def getKey():
	tty.setraw(sys.stdin.fileno())
	rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
	if rlist:
		key = sys.stdin.read(1)
	else:
		key = ''

	termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
	return key

def manual_shoot(key):

	if key == 'q' :
		print 'shoot button pressed'
		turret_thread.shoot_armour()


def auto_shoot(pitch_delta,yaw_delta,coord,y_bias,x_bias,Target_lock):
	if len(coord) > 0:
		target_coord = util.get_nearest_target(coord,y_bias,x_bias)
		height = target_coord[3]
		print pitch_delta,yaw_delta,height
		if MIN_PITCH_DELTA > abs(pitch_delta-pitch_bias) and MIN_YAW_DELTA > abs(yaw_delta-yaw_bias) and height > TARGET_MIN_HEIGHT:
			Target_lock +=1
			print"[!!!]Target within range, count = ",Target_lock
		else:
			print"Target not within range"
			print"pitch delta = ",abs(pitch_delta-pitch_bias)
			print"yaw delta = ",abs(yaw_delta-yaw_bias)
			Target_lock = 0

		if Target_lock >= MIN_CONSECUTIVE_TARGET_LOCKS:
			print"Target within range for more than x frames -> [SHOOT]"
			turret_thread.shoot_armour()
		else:
			robot_prop.shoot = 0

		return Target_lock


def draw_detection(img,coord):
	for x,y,width,height in coord:
		cv2.rectangle(img,(x-width//2,y-height//2),(x+width//2,y+height//2),(0,255,0),2)
	cv2.circle(img,(320,240),2,(255,0,0),-1) #draw image center
	cv2.imshow('img',img)
	cv2.waitKey(1)

def run(camera_thread,counter_coord,Target_lock):
	global pitch_bias,yaw_bias
	key = getKey()
	img = camera_thread.read()
	#cv2.imshow('img',img)
	#cv2.waitKey(1)
	coord = detection_mod.get_coord_from_detection(img)
	if len(coord) == 0 :
		counter_coord +=1
		print "No detection"
		#print counter_coord
		if counter_coord > 5:
			print "More than 5 frames without detection, No detection count :",counter_coord
			robot_prop.v1 = 0
			robot_prop.v2 = 0
	else:
		print "TARGET DETECTED"
		counter_coord = 0

	draw_detection(img, coord)

	# draw detection for debugging
	t_pitch = robot_prop.t_pitch
	t_yaw = robot_prop.t_yaw
	y_bias,x_bias = util.bias_to_pixel(pitch_bias,yaw_bias)
	pitch_delta,yaw_delta = util.get_delta(coord,y_bias,x_bias)

	if pitch_delta ==0 and yaw_delta ==0:
		robot_prop.v1 = t_pitch
		robot_prop.v2 = t_yaw
		#print t_pitch,t_yaw
		return counter_coord, Target_lock

	# change shoot function
	#manual_shoot(key)
	Target_lock = auto_shoot(pitch_delta,yaw_delta,coord,y_bias,x_bias,Target_lock)

	if key == '2' :
		pitch_bias-=5
		print"pitch bias = ",pitch_bias
	if key == '1' :
		pitch_bias+=5
		print"pitch bias = ",pitch_bias
	if key == '3' :
		yaw_bias-=5
		print"yaw bias = ",yaw_bias
	if key == '4' :
		yaw_bias+=5
		print"yaw bias = ",yaw_bias
	v1 = t_pitch + pitch_delta *pitch_weight - pitch_bias
	v2 = t_yaw + yaw_delta *yaw_weight - yaw_bias

	if abs(v1) >= 2000:
		v1 = 2000 * (v1/abs(v1))

	if abs(v2) >= 6000:	
		v2 = 6000 * (v2/abs(v2))
	print"pitch_delta = ",pitch_delta
	print"yaw_delta = ",yaw_delta
	robot_prop.v1 = v1
	robot_prop.v2 = v2
	return counter_coord,Target_lock
