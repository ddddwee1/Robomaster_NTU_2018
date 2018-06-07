import data_retriver
import robot_prop 
import time
import PID.PID

data_reader = data_retriver.data_reader_thread()
data_reader.start()

target_pitch = 20
target_yaw = 30

pid_pitch = PID.PID.PID(10,0,0.01)
pid_yaw = PID.PID.PID(10,0,0.01)

y = 100
while 1:
	t_pitch = robot_prop.t_pitch
	t_yaw = robot_prop.t_yaw
	error_pitch = target_pitch - t_pitch
	error_yaw = target_yaw - t_yaw

	v1 = pid_pitch.eval(error_pitch)
	v2 = pid_yaw.eval(error_yaw)

	robot_prop.v1 = v1
	robot_prop.v2 = v2
	print t_pitch, t_yaw
