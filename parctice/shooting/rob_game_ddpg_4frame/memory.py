from collections import deque 
import random

MAXLEN = 20000
MAX_PRIOR = 100
D = deque()
D2 = deque()

def push(inp):
	D.append(inp)
	if len(D)>MAXLEN:
		D.popleft()

def push_prior(inp):
	D2.append(inp)
	if len(D2)>MAX_PRIOR:
		D2.popleft()

def next_batch(BSIZE):
	if len(D2)>=2:
		p_batch = random.sample(D2,2)
	else:
		p_batch = []
	if BSIZE>len(D):
		return random.sample(D,len(D))+p_batch
	return random.sample(D,BSIZE-2)+p_batch