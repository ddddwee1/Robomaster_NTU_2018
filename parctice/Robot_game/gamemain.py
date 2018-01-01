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


reset()
done = False
end = False

FPSCLOCK = pygame.time.Clock()

def get_next_frame(act):

	action = np.argmax(act)

	reward = 0

	# if action==0:
	# 	act = [0,0,0,0,0,1,0,0]
	# elif action==1:
	# 	act = [0,0,0,0,0,0,1,0]
	# elif action==2:
	# 	act = [0,0,0,0,0,0,0,1]
	# 	reward = -0.1
	# else:
	# 	act = [0,0,0,0,0,0,0,0]

	if act[7]==1:
		bulletList.append(SpritesClass.Bullet(playerList.getList()[0]))

	pl = playerList.getList()
	pl[0].update(act)

	bulletList.updateAll()

	 # --- Collision Processing
	if bulletList.getList() != 0:
		for bullet1 in bulletList.getList():
			if playerList.getList() != 0:
				for player1 in playerList.getList():
					dx = abs(bullet1.getPosition()[0] - player1.getPosition()[0])
					dy = abs(bullet1.getPosition()[1] - player1.getPosition()[1])
					if dx**2 + dy**2 < (bullet1.getRadius() + player1.getRadius())**2:
						reward = 1.0

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

	pygame.display.flip()
	FPSCLOCK.tick(1)
	print(player1.getRotation())

	return [dx,dy],reward