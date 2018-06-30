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

#initial bias values
pitch_bias = 500
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
csv_file = open("/home/nvidia/bias_test.csv", 'a')

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


	if pitch_delta ==0 and yaw_delta ==0:
		continue
	#get height of bounding box
	height=util.get_height(coord)

	if key == '2' :
		print"Decreasing pitch bias.............."
		pitch_bias-=5

	if key == '1' :
		print"Increasing pitch bias.............."
		pitch_bias+=5

	if key == '3' :
		print"Decreasing pitch bias.............."
		yaw_bias-=5

	if key == '4' :
		print"Increasing yaw bias.............."
		yaw_bias+=5

	if key == 't' : #record as hitting the top edge
		csv_file.write(str(height))
		csv_file.write(';')
		csv_file.write(str(pitch_bias))
		csv_file.write(';')
		csv_file.write('top edge')
		csv_file.write('\n')
		print"[Write] height = ", height, '  pitch bias = ',pitch_bias, ' TOP EDGE!!'

	if key == 'b' : #record as hitting the btm edge
		csv_file.write(str(height))
		csv_file.write(';')
		csv_file.write(str(pitch_bias))
		csv_file.write(';')
		csv_file.write('btm edge')
		csv_file.write('\n')
		print"[Write] height = ", height, '  pitch bias = ',pitch_bias, ' BTM EDGE!!'

	if key == 's' :
		csv_file.close()
		csv_file = open("/home/nvidia/bias_test_upper_edge.csv", 'a')
		print"Saving to file................"

	print 'height' , height
	print 'pitch bias', pitch_bias
	print 'yaw bias', yaw_bias

	v1 = t_pitch + pitch_delta *1.0 - pitch_bias
	v2 = t_yaw + yaw_delta *1.0 - yaw_bias

	#enforce pitch limit
	if abs(v1) >= 2000:
		v1 = 2000
	#enforce yaw limit
	if abs(v2) >= 6000:
		v2 = 6000

	robot_prop.v1 = v1
	robot_prop.v2 = v2
	t2=time.time()
	t_e = t2-t1
	#print t_e
	#print t_pitch, t_yaw
