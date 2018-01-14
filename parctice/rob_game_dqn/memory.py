from collections import deque 
import random

MAXLEN = 20000
D = deque()

def push(inp):
	D.append(inp)
	if len(D)>MAXLEN:
		D.popleft()

def next_batch(BSIZE):
	return random.sample(D,BSIZE)