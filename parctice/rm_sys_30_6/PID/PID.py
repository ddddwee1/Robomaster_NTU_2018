import time 

class PID():
	def __init__(self,p,i,d):
		self.p = p
		self.i = i 
		self.d = d
		self.t = time.time()
		self.last_feedback = 0
		self.integ = 0

	def compute_integral(self,feedback,dt):
		self.integ += feedback * dt
		return self.i * self.integ

	def compute_diff(self,feedback,dt):
		dif = feedback - self.last_feedback
		dif = dif / dt
		return self.d * dif

	def compute_p(self,feedback):
		return self.p * feedback

	def eval(self,feedback):
		now = time.time()
		dt = now - self.t
		i = self.compute_integral(feedback,dt)
		d = self.compute_diff(feedback,dt)
		p = self.compute_p(feedback)
		#print ('D:',d, ' P:',p, 'i:',i)
		self.t = now 
		self.last_feedback = feedback
		return i+d+p 


