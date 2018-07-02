import numpy as np

class astar():
	def __init__(self,row,col):
		self.row = row
		self.col = col
		self.map = np.zeros([row,col],dtype=np.float32)
		self.testedmtx = np.zeros([row,col],dtype=np.float32)

	def set_obstacle(self,indices):
		for index in indices:
			r, c = index
			self.testedmtx[r][c] = 1
		#print(self.testedmtx)

	def remove_obstacle(self,indices):
		for index in indices:
			r,c = index
			self.testedmtx[r][c] = 0

	def compute_path(self,start,dest):
		obst_mtx = self.testedmtx.copy()
		self.nq = node_queue()
		# distance and position of current node
		curr_node = node(None, start, 0)
		position = start
		index = 0
		
		while not ((position[0]==dest[0]) and (position[1]==dest[1])):
			# print(self.testedmtx)
			# print(position)
			children = self.get_child(position)
			if (children == []):
				_, curr_node = self.nq.get()
				position = curr_node.coor
				continue
			for child in children:
				r, c = child
				self.testedmtx[r][c] = 1
			for child in children:
				dist_pass = curr_node.dist_pass
				new_dist_pass = dist_pass + self.compute_dist(position,child)
				new_node = node(curr_node, child, new_dist_pass)
				dist_buff = self.compute_dist(child, dest) + new_dist_pass
				self.nq.put([dist_buff, new_node])
			_, curr_node = self.nq.get()
			position = curr_node.coor
			
		path = curr_node.trace_back()
		# print(path)
		self.testedmtx = obst_mtx
		return path

	def compute_dist(self,pt1,pt2):
		return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

	def get_child(self,pnt):
		result = []
		r,c = pnt
		for i in range(3):
			for j in range(3):
				row = r - 1 + i
				col = c - 1 + j
				if (row <= self.row -1) and (row >= 0):
					if (col <= self.col -1) and (col >= 0):
						if not self.testedmtx[row][col]==1:
							result.append([row,col])
		# print("result=",result)
		return result

class node:
	def __init__(self, parent, coor, dist_pass):
		self.parent = parent
		self.coor = coor
		self.dist_pass = dist_pass
		
	def trace_back(self):
		path = []
		path.insert(0, self.coor)
		parent = self.parent
		
		while parent!=None:
			path.insert(0, parent.coor)
			parent = parent.parent
		return path
		
class node_queue:
	def __init__(self):
		self.queue = []
		
	def put(self, new_item):
		index = 0
		for item in self.queue:
			if item[0] < new_item[0]:
				index += 1;
			else:
				break
		self.queue.insert(index, new_item)
		
	def get(self):
		item = None
		if self.queue != []:
			item = self.queue[0]
			self.queue = self.queue[1:]
		return item

if __name__=='__main__':
	ddwe = astar(10,10)
	obstacle = [[0,2],[1,2],[2,2],[3,2],[4,2],[5,2],[3,4],[3,5],[3,6],[3,7]]
	ddwe.set_obstacle(obstacle)
	ddwe.compute_path([0,0],[0,9])
			
