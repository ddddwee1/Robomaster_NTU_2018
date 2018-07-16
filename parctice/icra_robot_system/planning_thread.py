import motion_main 
import robot_prop 
import threading
import time

class planning_thread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		time.sleep(0.5)
		while True:
			try:
				r2h,r2w,r2rot = robot_prop.get_r2()
				icramap.set_point_obst([r2h,r2w])
				motion_main.move_to(dst=[40,30],curr_pos=robot_prop.get_current_pos())
				icramap.remove_point_obst([r2h,r2w])
			except:
				robot_prop.lr = 0
				robot_prop.fw = 0
				robot_prop.rot = 0
				
