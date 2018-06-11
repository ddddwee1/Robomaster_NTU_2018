import data_retriver
import robot_prop 
import time
import util
import detection_mod
from camera_module import camera_thread
import cv2
import PID.ema

data_reader = data_retriver.data_reader_thread()
data_reader.start()

camera_thread = camera_thread()
camera_thread.start()

EMA_pitch = PID.ema.EMA(0.99)
EMA_yaw  = PID.ema.EMA(0.99)

counter_detection = 0
counter_shoot = 0

while True:
	t1=time.time()
	img = camera_thread.read()
	coord = detection_mod.get_coord_from_detection(img)
	#print img.shape

	#if len(coord) == 0 and counter_shoot >=10:
	#	robot_prop.shoot = 0
	#	counter_shoot = 0
		#print 'a' ,counter
	#else:
	#	robot_prop.shoot = 2
	#	counter_shoot +=1
		#print 'b' , counter

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

	#EMA_pitch_delta  = EMA_pitch.update(pitch_delta)
	#EMA_yaw_delta  = EMA_pitch.update(yaw_delta)


	print 'p_d', pitch_delta
	print 'y_d', yaw_delta

	v1 = t_pitch + pitch_delta *0.9
	v2 = t_yaw + yaw_delta *1.4

	if abs(v1) >= 2000:
		v1 = 2000

	if abs(v2) >= 6000:
		v2 = 6000


	robot_prop.v1 = v1
	robot_prop.v2 = v2
	t2=time.time()
	t_e = t2-t1
	#print t_e
	#print t_pitch, t_yaw
