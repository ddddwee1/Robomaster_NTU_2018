from socket import *
import threading 
import numpy as np 
import robot_prop
import util
import time 

class network_thread(threading.Thread):
	def __init__(self,serverip):
		self.socket = socket(AF_INET, SOCK_STREAM)
		self.socket.connect((serverip,8080))
		threading.Thread.__init__(self)

	def run(self):
		conn = self.socket
		while True:
			time.sleep(0.05)
			datapack = b'\xaa\xcc'
			h,w,ang = robot_prop.get_current_pos()
			datapack += util.encode_whang(w,h,ang)
			conn.send(datapack)

			data = conn.recv(1)
			if data==b'\xaa':
				data = conn.recv(1)
				if data == b'\xcc':
					data = conn.recv(6)
					w,h,ang = util.decode_whang(data)
					robot_prop.r2h = h
					robot_prop.r2w = w 
					robot_prop.r2rot = ang 
					