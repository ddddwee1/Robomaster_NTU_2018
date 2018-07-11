import numpy as np 
from math import sqrt
import cv2 

def bias_to_pixel(pitch_bias,yaw_bias):

	return int(float(pitch_bias*240/2800)),int(float(yaw_bias*320/2800))

def get_nearest_target(coord,y_bias,x_bias):
	#get nearest target to image center
	distance = [(x-320+x_bias)*(y-240+y_bias) for x,y,w,h in coord]
	min_index = np.argmax(-np.array(distance))
	target_coord = coord[min_index]
	return target_coord

def get_delta(coord,y_bias,x_bias):
	if len(coord) == 0:
		return 0,0
	target_coord = get_nearest_target(coord,y_bias,x_bias)
	x,y,w,h = target_coord
	# do process on x,y,w,h
	return int(float(y-240)*2800/240),int(float(x-320)*2800/320)

def get_delta_buf(x,y):

	# do process on x,y,w,h
	return int(float(y-240)*2800/240),int(float(x-320)*2800/320)


