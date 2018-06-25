import rune_recog_template
import conv
import cv2
import time
import numpy as np
import data_retriver
import robot_prop
import time
import util
import detection_mod
from camera_module import camera_thread
import cv2
import sys, select, termios, tty
import math
import digit_detection

def getKey():
	tty.setraw(sys.stdin.fileno())
	rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
	if rlist:
		key = sys.stdin.read(1)
	else:
		key = ''

	termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
	return key


if __name__ == '__main__'
    #Parameters for turret control
    KNOWN_DISTANCE = 2.0
    KNOWN_WIDTH = 40.0

    #Threads required for turret control
    data_reader = data_retriver.data_reader_thread()
    data_reader.start()
    camera_thread = camera_thread()
    camera_thread.start()

    #Parameters for detection
    WHITE = (255,255,255)
    BLACK = (0,0,0)
    image_width = 1024
    image_height = 768
    # counter_shoot = 0

    #Detection intitialisation
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FPS,60)
    cap.set(3, image_width)
    cap.set(4, image_height)
    cap.set(14, 0.0)  #exposure
    cap.set(10, 0.05) #brightness

    #Keyboard Initialisation
    settings = termios.tcgetattr(sys.stdin)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    key = None

    while True:
        _,image = cap.read()
        key = int(getKey())
        digit_coords = get_digit_coords(image, DEBUG = True)
        # target = None
        # v1 = 0
        # v2 = 0
        # if key is not None and len(digit_coords) == 0:
        #     print"No handwritten digits detected!"
        # else:
        #     #current pitch and yaw
        #     t_pitch = robot_prop.t_pitch
        #     t_yaw = robot_prop.t_yaw

        #     # pitch_delta, yaw_delta = get_pitch_and_yaw_delta(digit_coords[key])
        #     x_delta = digit_coords[0]-320
        #     y_delta = digit_coords[1]-240
        #     pitch_delta = y_delta**3 * 100
        #     yaw_delta = x_delta**3 * -100
        #     v2 = robot_prop.t_pitch + x_delta
        #     v1 = robot_prop.t_yaw + y_deltas

        # #Enforce joint limit
        # if abs(v1) >= 2000:
        #     v1 = 2000
        # if abs(v2) >= 6000:
        #     v2 = 6000

        # #Adjust pitch and yaw angle
        # robot_prop.v1 = v1 #pitch
        # robot_prop.v2 = v2 #yaw