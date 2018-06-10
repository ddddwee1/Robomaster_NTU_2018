import data_retriver
import robot_prop 
import time
import util
import detection_mod
from camera_module import camera_thread
import cv2

data_reader = data_retriver.data_reader_thread()
data_reader.start()

camera_thread = camera_thread()
camera_thread.start()

dst_pitch = -20
dst_yaw = 60

while 1:
	img = camera_thread.read()
	coord = detection_mod.get_coord_from_detection(img)
	#print img.shape
	
	for x,y,width,height in coord:
		cv2.rectangle(img,(x-width//2,y-height//2),(x+width//2,y+height//2),(0,255,0),2)
	cv2.imshow('img',img)
	cv2.waitKey(1)
	#print coord

	t_pitch = robot_prop.t_pitch
	t_yaw = robot_prop.t_yaw

	yaw_delta,pitch_delta = util.get_delta(coord)

	if abs(yaw_delta) >= 6000:
		yaw_delta = 6000

	if abs(pitch_delta) >= 2000:
		yaw_delta = 2000

	print 'p_d', pitch_delta
	print 'y_d', yaw_delta
	
	if pitch_delta ==0 and yaw_delta ==0:
		continue

	v1 = t_pitch + pitch_delta*0.5
	v2 = t_yaw + yaw_delta*0.5

	robot_prop.v1 = v1
	robot_prop.v2 = v2
	print t_pitch, t_yaw
