import numpy as np
import random

class data_reader():
	def __init__(self):
		f = open('trainlist.txt')
		data = []
		for i in f:
			i = i.strip()
			f2 = open(i)
			res = self.readFile(f2.read(),5)
			f2.close()
			data += res 
		self.val_data = data[:1000]
		self.data = data[1000:]

	def readFile(self,st,n):
		st = st.split('\n')[:-1]
		data = []
		env = []
		for i in range(len(st)):
			row = st[i].split(',')

			row = [float(i) for i in row]
			if i==0:
				for _ in range(n):
					env.append(row[0])
					env.append(row[1])
			else:
				env = env[2:]
				env.append(row[0])
				env.append(row[1])
			action = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
			# very stupid to get the action array
			action[1+int(row[2])] = 1.
			action[4+int(row[3])] = 1.
			action[7+int(row[4])] = 1.
			action[9] = row[5]

			Q = [row[-1]]
			data.append([env,Q,action])
		return data

	def next_batch(self,bsize):
		return random.sample(self.data,bsize)

	def get_val(self):
		return self.val_data