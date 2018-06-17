import PID 

class PID2d():
	def __init__(self,p, i, d):
		self.pidx = PID.PID(p, i, d)
		self.pidy = PID.PID(p, i, d)

	def eval(self,offset):
		x,y= offset[0], offset[1]
		result_x = self.pidx.eval(x)
		result_y = self.pidy.eval(y)
		return result_x, result_y
