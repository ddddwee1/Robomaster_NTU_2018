import robot_prop 
import data_retriver
import detection_main 
import motion_main 
import control_test
from mp_util import icramap

data_reader = data_retriver.data_reader_thread()
data_reader.start()

detection = detection_main.detection_non_thread()

for i in range(100):
	detection.get_detection()
	r2h,r2w,r2rot = robot_prop.get_r2()
	icramap.set_point_obst([r2h,r2w])
	motion_main.move_to(dst=[40,30],curr_pos=robot_prop.get_current_pos())
	icramap.remove_point_obst([r2h,r2w])
	# disable read while writing 
	robot_prop.port_writing = True
	v1,v2,lr,fw,rot,shoot = robot_prop.get_order()
	control_test.send_order(v1,v2,lr,fw,rot,shoot)
	robot_prop.port_writing = False