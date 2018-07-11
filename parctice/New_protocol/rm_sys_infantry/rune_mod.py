import cv2
import time
import numpy as np
import data_retriver
import robot_prop
import util
from camera_module import camera_thread
from turret_module import turret_thread
from rune_module import rune_shooting_logic

turret_thread = turret_thread()
turret_thread.start()

pitch_bias = 730
yaw_bias = 0

scr_7seg_index = 0
saved_7seg_raw = -1
saved_numbers_9boxes = -1
num_9boxes_index = -1

whether_shooted = False

def run(camera_thread):

	global pitch_bias
	global yaw_bias

	global scr_7seg_index
	global saved_7seg_raw
	global saved_numbers_9boxes
	global num_9boxes_index

	global whether_shooted

	image = camera_thread.read()

	if image is None:
		return
	print ('robot_prop.time_remain: ' , robot_prop.time_remain)
#	if robot_prop.time_remain > 240:
#		bigbuff = False
#	else:
#		bigbuff = True
	bigbuff = True

	handwritten_coord, saved_numbers_9boxes , saved_7seg_raw , scr_7seg_index , num_9boxes_index, whether_shooted = rune_shooting_logic.rune_shooting(image,saved_numbers_9boxes , saved_7seg_raw , scr_7seg_index , num_9boxes_index, whether_shooted,bigbuff)

	#print 'abc', handwritten_coord , saved_numbers_9boxes , saved_7seg_raw , scr_7seg_index , num_9boxes_index, whether_shooted

	if len(handwritten_coord)==1:
		return

	shoot_coord = handwritten_coord[num_9boxes_index]
	x,y = shoot_coord 
	t_pitch = robot_prop.t_pitch
	t_yaw = robot_prop.t_yaw

	pitch_delta,yaw_delta = util.get_delta_buf(x,y)

	if pitch_delta ==0 and yaw_delta ==0:
		return

	v1 = t_pitch + pitch_delta *0.9 - pitch_bias
	v2 = t_yaw + yaw_delta *1.4 - yaw_bias

	robot_prop.v1 = v1
	robot_prop.v2 = v2
	#time.sleep(0.1)
	if whether_shooted == False:
		turret_thread.shoot()
		whether_shooted = True
	#print scr_7seg_raw, handwritten_num, Flaming_digit
