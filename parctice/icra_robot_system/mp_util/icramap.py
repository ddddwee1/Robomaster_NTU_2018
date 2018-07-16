import Astar

def get_obs_list(inp):
	res = []
	for item in inp:
		r0,c0 = item[0]
		r1,c1 = item[1]
		for i in range(max(0,r0-2),min(r1+3,50)):
			for j in range(max(0,c0-2),min(c1+3,80)):
				res.append([i,j])
	return res 

a_star = Astar.astar(50,80)

# initialize map
obs_list = get_obs_list([
	[[15,18],[27,21]],
	[[37,12],[40,20]],
	[[22,0 ],[25,8 ]],
	[[0 ,31],[20,34]],
	[[30,46],[50,49]],
	[[23,59],[35,62]],
	[[10,60],[13,68]],
	[[25,72],[28,80]]
	])

a_star.set_obstacle(obs_list)

# do avoid obstacle by setting obstacle in map 
def set_point_obst(p0):
	a_star.set_obstacle([p0])

def remove_point_obst(p0):
	a_star.remove_obstacle([p0])

def get_next_point(p0,p1):
	path = a_star.compute_path(p0,p1)
	return path[1]
