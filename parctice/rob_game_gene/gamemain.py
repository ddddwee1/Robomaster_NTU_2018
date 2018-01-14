import pygame 
from pygame.locals import *
import Display
import SpritesClass
import numpy as np 

pygame.init()

display = Display.Display('games')
screen = display.getScreen()

def reset():
	global playerList,bulletList
	playerList = SpritesClass.PlayerList()
	bulletList = SpritesClass.BulletList()

	player1 = SpritesClass.Player(250,250,0,1)
	player2 = SpritesClass.Player(450,300,180,2)

	playerList.append(player1)
	playerList.append(player2)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

reset()
done = False
end = False

FPSCLOCK = pygame.time.Clock()

def get_next_frame(act):
	reward = 0

	if act[0]<-0.5:
		action = [0,0,0,0,0,1,0,0]
	elif act[0]>0.5:
		action = [0,0,0,0,0,0,1,0]
	else:
		action = [0,0,0,0,0,0,0,0]



	# if act[1]>0:
	# 	action[-1] = 1

	if action[7]==1:
		bulletList.append(SpritesClass.Bullet(playerList.getList()[0]))

	pl = playerList.getList()
	pl[0].update(action)

	bulletList.updateAll()

	 # --- Collision Processing
	if bulletList.getList() != 0:
		for bullet1 in bulletList.getList():
			if playerList.getList() != 0:
				for player1 in playerList.getList():
					dx = abs(bullet1.getPosition()[0] - player1.getPosition()[0])
					dy = abs(bullet1.getPosition()[1] - player1.getPosition()[1])
					if dx**2 + dy**2 < (bullet1.getRadius() + player1.getRadius())**2:
						print('Hit ')
						# reward += 10.0

	if playerList.getList() != 0:
		for player1 in playerList.getList():
			for player2 in playerList.getList():
				if player1 != player2:
					dx = abs(player1.getPosition()[0] - player2.getPosition()[0])
					dy = abs(player1.getPosition()[1] - player2.getPosition()[1])
					if dx**2 + dy**2 < (player1.getRadius() + player2.getRadius())**2:
						player1.cancelUpdate()
						player2.cancelUpdate()

	screen.fill((255,255,255))
	player1 = playerList.getList()[0]
	player2 = playerList.getList()[1]
	display.drawAllPlayers(playerList)
	display.drawAllBullets(bulletList)
	display.drawLaserSight(player1)
	display.drawLaserSight(player2)

	pos0 = player1.getPosition()
	pos1 = player2.getPosition()

	dx = pos1[0]-pos0[0]
	dy = pos1[1]-pos0[1]

	r, rad = cart2pol(-dy,dx)

	rdis = rad-player1.getRotation()*np.pi/180.

	if rdis>np.pi:
		rdis -= np.pi*2
	if rdis<-np.pi:
		rdis += np.pi*2

	reward = reward + (1. - abs(rdis))
	reward = reward*10
	reward = reward//5
	reward = - reward

	pygame.display.flip()
	FPSCLOCK.tick(100)
	# print(rad,player1.getRotation()*np.pi/180.)
	# print(player1.getRotation())
	# print(reward)
	# print(rdis)
	# print(player1.getRotation()*np.pi/180.)
	# print(rad)

	return r/300,rdis,reward