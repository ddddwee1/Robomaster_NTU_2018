class EMA():
	def __init__(self,alpha):
		self.alpha = alpha
		self.value = None
	def update(self,value):
		if self.value is None:
			self.value = value
		else:
			self.value = self.alpha * self.value + (1 - self.alpha)*value
		return self.value
	def get_value(self):
		return self.value
