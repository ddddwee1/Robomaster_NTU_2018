import robot_prop 
import data_retriver
import detection_main 
import motion_main 
import control_test
from mp_util import icramap
import time

data_reader = data_retriver.data_reader_thread()
data_reader.start()

detection = detection_main.detection_non_thread()

for i in range(100000):
	try:
		motion_main.move_to(dst=[25,30],curr_pos=robot_prop.get_current_pos())
	except:
		print 'planning wrong'
		robot_prop.lr = 0 
		robot_prop.fw = 0
		robot_prop.rot = 0 
	detection.get_detection()
	robot_prop.port_writing = True
	v1,v2,lr,fw,rot,shoot = robot_prop.get_order()
	control_test.send_order(v1,v2,lr,fw,rot,shoot)
	robot_prop.port_writing = False	
