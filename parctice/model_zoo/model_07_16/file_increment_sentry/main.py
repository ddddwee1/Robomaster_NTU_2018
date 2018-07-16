import data_retriver
import robot_prop 
import time
import util
import detection_mod
from camera_module import camera_thread_0, camera_thread_1, camera_thread_2
from turret_module import turret_thread
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

turret_thread = turret_thread()
turret_thread.start()

#Parameters
pitch_bias = 250
yaw_bias = 80
TARGET_MIN_HEIGHT = 12 #Acceptable minimum height of the target before the turret shoots at it
MIN_PITCH_DELTA = 200
MIN_YAW_DELTA = 100
MIN_CONSECUTIVE_TARGET_LOCKS = 2
pitch_weight = 1.3
yaw_weight = 2.0

turret_cam_detected = False
y1_coefficient = 100 #TODO
x1_coefficient = 100 #TODO
y2_coefficient = 100 #TODO
x2_coefficient = 100 #TODO

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
			#print"[!!!]Target within range, count = ",Target_lock
		else:
			#print"Target not within range"
			#print"pitch delta = ",abs(pitch_delta-pitch_bias)
			#print"yaw delta = ",abs(yaw_delta-yaw_bias)
			Target_lock = 0

		if Target_lock >= MIN_CONSECUTIVE_TARGET_LOCKS:
			#print"Target within range for more than x frames -> [SHOOT]"
			turret_thread.shoot_armour()
		else:
			robot_prop.shoot = 0

		return Target_lock

def draw_detection(img,coord):
	for x,y,width,height in coord:
		cv2.rectangle(img,(x-width//2,y-height//2),(x+width//2,y+height//2),(0,255,0),2)
	cv2.imshow('img',img)
	cv2.waitKey(1)

def turret_no_detected():
	img_1 = camera_thread_1.read()
	img_2 = camera_thread_2.read()
	coord_1 = detection_mod.get_coord_from_detection_small(img_1)
	coord_2 = detection_mod.get_coord_from_detection_small(img_2)
	if coord_1 == [] and coord_2 == []:
		robot_prop.mode = 0
		return 'none' , None
	elif coord_1 != []:
		target_coord_1 = util.get_nearest_target(coord_1,0,0)
		robot_prop.mode = 1
		return 'front' , target_coord_1
	else:
		target_coord_2 = util.get_nearest_target(coord_2,0,0)
		robot_prop.mode = 1
		return 'back' , target_coord_2

while True:

	key = getKey()
	img = camera_thread.read()
	coord = detection_mod.get_coord_from_detection(img)
	if len(coord) == 0 :
		counter_coord +=1
		#print "No detection"
		#print counter_coord
		if counter_coord > 3:
			#print "More than 5 frames without detection, No detection count :",counter_coord
			robot_prop.v1 = 0
			robot_prop.v2 = 9000
			robot_prop.mode = 0

	else:
		#print "TARGET DETECTED"
		robot_prop.mode = 1
		counter_coord = 0


	if robot_prop.mode == 0:
		base_detect,target_coord_1_2 = turret_no_detected()
		if base_detect == 'front':
			x_1,y_1,w_1,h_1 = target_coord_1_2
			robot_prop.v1 = 0 - y1_coefficient*(y_1-240)
			robot_prop.v2 = -9000 + x1_coefficient*(x_1-320)
		elif base_detect == 'back':
			x_2,y_2,w_2,h_2 = target_coord_1_2
			robot_prop.v1 = 0 - y2_coefficient*(y_2-240)
			robot_prop.v2 = 9000 + x2_coefficient*(x_2-320)
		continue

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
		continue

	# change shoot function
	manual_shoot(key)
	#Target_lock = auto_shoot(pitch_delta,yaw_delta,coord,y_bias,x_bias,Target_lock)

	if key == '2' :
		pitch_bias-=5

	if key == '1' :
		pitch_bias+=5

	if key == '3' :
		yaw_bias-=5

	if key == '4' :
		yaw_bias+=5

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

