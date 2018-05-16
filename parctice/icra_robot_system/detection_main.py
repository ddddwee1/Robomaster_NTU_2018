from PID.PID2d import PID2d
import camera_module
import detection_mod
import util
import threading
import robot_prop 

class detection_class(threading.Thread):
	def __init__(self):
		# initialize all instances
		self.pid = PID2d(2.28,0.000,-0.0000228)
		self.camera_thread = camera_module.camera_thread()
		self.v = [0.,0.]
		self.shoot = 0
		threading.Thread.__init__(self)

	def run(self):
		print('Running detection module')
		self.camera_thread.start()
		while True:
			img = self.camera_thread.read()
			coord = detection_mod.get_coord_from_detection(img)
			v1,v2,shoot = util.get_v_and_shoot(coord,self.pid)
			self.v = [v1,v2]
			self.shoot = shoot

	def get_attr(self):
		return self.v,self.shoot

class detection_non_thread():
	def __init__(self):
		self.pid = PID2d(2.28,0.000,-0.0000228)
		self.camera_thread = camera_module.camera_thread()
		self.camera_thread.start()

	def get_detection(self):
		img = self.camera_thread.read()
		coord = detection_mod.get_coord_from_detection(img)
		v1,v2,shoot = util.get_v_and_shoot(coord,self.pid)
		v = [v1,v2]
		shoot = shoot
		robot_prop.v1 = v1 
		robot_prop.v2 = v2 
		robot_prop.shoot = shoot
		return v,shoot