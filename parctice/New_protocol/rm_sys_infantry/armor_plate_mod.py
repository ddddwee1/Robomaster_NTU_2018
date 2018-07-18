import robot_prop
import time
import util
import detection_mod
import cv2
import sys, select, termios, tty


settings = termios.tcgetattr(sys.stdin)
termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


PITCH_BIAS= 600
YAW_BIAS= 0
TARGET_MIN_HEIGHT= 12 #Acceptable minimum height of the target before the turret shoots at it
MIN_PITCH_DELTA= 130  #Acceptable minimum height of the target before the turret shoots at it
MIN_YAW_DELTA= 150     #Acceptable minimum height of the target before the turret shoots at it
MIN_CONSECUTIVE_TARGET_LOCKS= 2
PITCH_WEIGHT= 1.15
YAW_WEIGHT= 1.5
mode = 'auto'

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
	global PITCH_BIAS,YAW_BIAS
	if key == '2' :
		PITCH_BIAS -= 5
		print"Minus pitch_bias.------------------ pitch bias = ",PITCH_BIAS
	if key == '1' :
		PITCH_BIAS+=5
		print"Add pitch_bias.------------------ pitch bias = ",PITCH_BIAS
	if key == '3' :
		YAW_BIAS-=5
		print"Minus pitch_bias.------------------ yaw bias = ",YAW_BIAS
	if key == '4' :
		YAW_BIAS+=5
		print"Add pitch_bias. ------------------yaw bias = ",YAW_BIAS
	if key == 'q' :
		print '---[SHOOT]---'
		robot_prop.shoot = 1
		time.sleep(0.1)
		robot_prop.shoot = 0


def auto_shoot(pitch_delta,yaw_delta,coord,y_bias,x_bias,target_lock):
	target_coord = util.get_nearest_target(coord,y_bias,x_bias)
	height = target_coord[3]
	if MIN_PITCH_DELTA > abs(pitch_delta-PITCH_BIAS) and MIN_YAW_DELTA > abs(yaw_delta-YAW_BIAS) and height > TARGET_MIN_HEIGHT:
		target_lock +=1
		print"[!!!]Target within range, count = ",target_lock
	else:
#			print"Target not within range ~~"
		if abs(pitch_delta - PITCH_BIAS) > MIN_PITCH_DELTA:
			print "pitch delta not within range"
			print "pitch delta = ",pitch_delta
			print"abs(pitch delta - pitch bias) = ",abs(pitch_delta-PITCH_BIAS)
		if abs(yaw_delta - YAW_BIAS) > MIN_YAW_DELTA:
			print "yaw delta not within range"
			print "yaw_delta = ",yaw_delta
			print"abs(yaw delta - yaw bias) = ",abs(yaw_delta-YAW_BIAS)
		if height < TARGET_MIN_HEIGHT:
			print "target min height not met"
			print "height = ",height
		target_lock = 0
	if target_lock >= MIN_CONSECUTIVE_TARGET_LOCKS:
		print"Target within range for more than {} frames -> [SHOOT]".format(MIN_CONSECUTIVE_TARGET_LOCKS)
		robot_prop.shoot = 1
	else:
		robot_prop.shoot = 0

	return target_lock


def draw_detection(img,coord):
	for x,y,width,height in coord:
		cv2.rectangle(img,(x-width//2,y-height//2),(x+width//2,y+height//2),(0,255,0),2)
	cv2.circle(img,(320,240),2,(255,0,0),-1) #draw image center
	cv2.imshow('img',img)
	cv2.waitKey(1)

def run(camera_thread,target_lock):
	#t1 = time.time()
	global PITCH_BIAS,YAW_BIAS,first_time_no_detection, stop_pitch, stop_yaw
	img = camera_thread.read()
	cv2.imshow('img',img)
	coord = detection_mod.get_coord_from_detection(img)
	current_pitch = robot_prop.t_pitch
	current_yaw = robot_prop.t_yaw


	#Armour plate(s) detected
	if len(coord) != 0:
		first_time_no_detection = True
		draw_detection(img,coord)
		y_bias, x_bias = util.bias_to_pixel(PITCH_BIAS,YAW_BIAS)
		pitch_delta, yaw_delta = util.get_delta(coord,y_bias,x_bias)
		pitch_goal = current_pitch + pitch_delta *PITCH_WEIGHT - PITCH_BIAS
		yaw_goal = current_yaw + yaw_delta*YAW_WEIGHT - YAW_BIAS
		robot_prop.v1 = pitch_goal
		robot_prop.v2 = yaw_goal
		if mode == 'auto':
			auto_shoot(pitch_delta, yaw_delta, coord, y_bias, x_bias, target_lock)
		elif mode == 'manual':
			key = getKey()
			manual_shoot(key)


	#computation_time = time.time() - t1
	#print "computational time = ",computation_time
	return target_lock
