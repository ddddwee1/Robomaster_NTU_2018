import digit_detection
import cv2
import time
import numpy as np
import data_retriver
from camera_module import camera_thread
import robot_prop
import util
import sys, select, termios, tty
import time
from turret_module import turret_thread


data_reader = data_retriver.data_reader_thread()
data_reader.start()

camera_thread = camera_thread()
camera_thread.start()

turret_thread = turret_thread()
turret_thread.start()

image_width = 640
image_height = 480

scr_7seg_index = 0
saved_7seg_raw = -1
saved_numbers_handwritten = -1
handwritten_num_index = -1

counter_detection = 0
counter_shoot = 0

whether_shooted = False

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

while True:

	new_7seg = False
	key = getKey()
	image = camera_thread.read()
	if image is None:
		break
	try:
		coord,scr_7seg_raw, handwritten_num, Flaming_digit = digit_detection.get_digits2(image)
	except:
		time.sleep(0.1)
		continue

	if len(handwritten_num) == 1 and len(Flaming_digit) == 1 :
		cv2.imshow('',image)
		cv2.waitKey(1)
		print ("Wrong number")
		continue

	x1 = (coord[2][0] + coord[2][2] + coord[3][0] +coord[3][2] - coord[0][0] - coord[1][0])//4 +  (coord[0][0] + coord[1][0]) //2
	y1 = (coord[1][1] + coord[1][3] + coord[3][1] +coord[3][3] - coord[0][1] - coord[2][1])//4 + (coord[0][1] + coord[2][1]) //2

	x0 = (x1 - coord[0][0])//2 + coord[0][0]
	y0 = (coord[0][1] +coord[0][3] + coord[2][1] + coord[0][3])//2

	x2 = (coord[2][0] + coord[2][2] - x1)//2 + x1
	y2 = (coord[1][1] + coord[3][1])//2

	radius = (coord[0][3]) //4

	handwritten_coord = [[x0,y0],[x1,y0],[x2,y0],[x0,y1],[x1,y1],[x2,y1],[x0,y2],[x1,y2],[x2,y2]]
	for x,y in handwritten_coord:
		cv2.circle(image,(x,y),radius, (0,0,255), 4)

	cv2.imshow('',image)
	cv2.waitKey(1)

	for raw_7seg in scr_7seg_raw:
		if raw_7seg == 10:
			checked_7seg_raw = False
			break
		else: 
			checked_7seg_raw = True

	numbers_7seg = scr_7seg_raw[0]*10000+scr_7seg_raw[1]*1000+scr_7seg_raw[2]*100+scr_7seg_raw[3]*10+scr_7seg_raw[4]

	if saved_7seg_raw != numbers_7seg and checked_7seg_raw == True:
		saved_7seg_raw = numbers_7seg 
		scr_7seg_index = 0
		new_7seg = True
		whether_shooted = False
		print 'abc'

	if saved_7seg_raw == -1 or saved_7seg_raw == numbers_7seg and checked_7seg_raw == True:
		num_7seg = scr_7seg_raw[scr_7seg_index]
		saved_7seg_raw = numbers_7seg
		if scr_7seg_index == 0:
			prev_num_7seg = num_7seg
		else:
			prev_num_7seg = scr_7seg_raw[scr_7seg_index-1]
		#print ('a',num_7seg)

	numbers_handwritten = handwritten_num[0]*10**8+handwritten_num[1]*10**7+handwritten_num[2]*10**6+handwritten_num[3]*10**5+handwritten_num[4]*10**4+handwritten_num[5]*10**3+handwritten_num[6]*10**2+handwritten_num[7]*10**1+handwritten_num[8]

	if saved_numbers_handwritten == -1 or saved_numbers_handwritten !=  numbers_handwritten or new_7seg == True:
		saved_numbers_handwritten = numbers_handwritten
		scr_7seg_index += 1
		whether_shooted = False
		for index in range(len(handwritten_num)):
			if num_7seg == handwritten_num[index]:
				handwritten_num_index = index
				print ('b',prev_num_7seg,handwritten_num_index)

	else:
		if handwritten_num_index != -1:
			print ('b',prev_num_7seg,handwritten_num_index)

	if scr_7seg_index >=5:
		scr_7seg_index = 0


	shoot_coord = handwritten_coord[handwritten_num_index]
	x,y = shoot_coord 
	t_pitch = robot_prop.t_pitch
	t_yaw = robot_prop.t_yaw

	pitch_delta,yaw_delta = util.get_delta_buf(x,y)

	if pitch_delta ==0 and yaw_delta ==0:
		continue
	pitch_bias = 700
	v1 = t_pitch + pitch_delta *1.0 - pitch_bias
	v2 = t_yaw + yaw_delta *1.4

	robot_prop.v1 = v1
	robot_prop.v2 = v2
	#time.sleep(0.1)
	if whether_shooted == False:
		turret_thread.shoot()
		whether_shooted = True
	#print scr_7seg_raw, handwritten_num, Flaming_digit
