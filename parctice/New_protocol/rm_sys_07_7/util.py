import numpy as np 
from math import sqrt
import cv2 

def pitchbias_to_ypixel(pitch_bias):

	return int(float(pitch_bias*240/3000))

def get_nearest_target(coord,y_bias):
	#get nearest target to image center
	distance = [(x-320)*(y-240+y_bias) for x,y,w,h in coord]
	min_index = np.argmax(-np.array(distance))
	target_coord = coord[min_index]
	return target_coord

def get_delta(coord,y_bias = 0):
	if len(coord) == 0:
		return 0,0
	target_coord = get_nearest_target(coord,y_bias)
	x,y,w,h = target_coord
	# do process on x,y,w,h
	return int(float(y-240)*3000/240),int(float(x-320)*3000/320)

def get_delta_buf(x,y):

	# do process on x,y,w,h
	return int(float(y-240)*3000/240),int(float(x-320)*3000/320)


