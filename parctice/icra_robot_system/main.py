import robot_prop 
import data_retriver
import detection_main 
import motion_main 

data_reader = data_retriver.data_reader_thread()
data_reader.start()

detection = detection_main.detection_non_thread()

while 1:
	try:
		motion_main.move_to(dst=[25,30],curr_pos=robot_prop.get_current_pos())
	except:
		print 'planning wrong'
		robot_prop.lr = 0 
		robot_prop.fw = 0
		robot_prop.rot = 0 
	# The pitch, yaw, and movement are modified in detection object
	detection.get_detection()
