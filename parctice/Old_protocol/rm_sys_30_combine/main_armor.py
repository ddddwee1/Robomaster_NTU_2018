import data_retriver
import robot_prop 
import time
import util
import detection_mod
from camera_module import camera_thread
import cv2
import sys, select, termios, tty
import math

KNOWN_DISTANCE = 2.0
KNOWN_WIDTH = 40.0

data_reader = data_retriver.data_reader_thread()
data_reader.start()

camera_thread = camera_thread()
camera_thread.start()

counter_detection = 0
counter_shoot = 0

pitch_bias = 920
yaw_bias = 0

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
	key = getKey()
	t1=time.time()
	img = camera_thread.read()
	coord = detection_mod.get_coord_from_detection(img)
	#print img.shape

	if len(coord) == 0 and counter_shoot >=10:
		robot_prop.shoot = 0
		counter_shoot = 0
		#print 'a' ,counter
	else:
		if key == 'q' :
			print 'a'
			robot_prop.shoot = 2
			counter_shoot +=1
			#print 'b' , counter

		else: 
			robot_prop.shoot = 0

	for x,y,width,height in coord:
		cv2.rectangle(img,(x-width//2,y-height//2),(x+width//2,y+height//2),(0,255,0),2)
	cv2.imshow('img',img)
	cv2.waitKey(1)
	#print coord


	t_pitch = robot_prop.t_pitch
	t_yaw = robot_prop.t_yaw

	pitch_delta,yaw_delta = util.get_delta(coord)
#	if abs(pitch_delta) < 200:
#		pitch_delta = 0
#	else:
#		print 'p_d', pitch_delta, coord
#
#	if abs(yaw_delta) < 200:
#		yaw_delta = 0
#	else:
#		print 'y_d', yaw_delta,coord


	if pitch_delta ==0 and yaw_delta ==0:
		continue

	height=util.get_height(coord)

	if key == '2' :
		pitch_bias-=5

	if key == '1' :
		pitch_bias+=5

	if key == '3' :
		yaw_bias-=5

	if key == '4' :
		yaw_bias+=5


	print 'height' , height
	print 'pitch bias', pitch_bias
	print 'yaw bias', yaw_bias

	v1 = t_pitch + pitch_delta *1.0 - pitch_bias
	v2 = t_yaw + yaw_delta *1.0 - yaw_bias


	if abs(v1) >= 2000:
		v1 = 2000 * (v1/abs(v1))

	if abs(v2) >= 6000:
		v2 = 6000 * (v2/abs(v2))

	robot_prop.v1 = v1
	robot_prop.v2 = v2
	t2=time.time()
	t_e = t2-t1
	#print t_e
	#print t_pitch, t_yaw
