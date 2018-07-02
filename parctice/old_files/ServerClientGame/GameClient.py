import pygame
from pygame.locals import *
import random, math, sys
import Display
import Functions
import SpritesClass
import InputHandling
from socket import *
import time, random

# Setting up the port
Port, HostIP = 27400, "192.168.1.101"

# HostIP = input("Please input the Host/Server IP Address : ")
# Port = int(input("Please input the game's port on the Host/Server :"))

# Connecting to server
with socket(AF_INET, SOCK_STREAM) as soc:
    soc.connect((HostIP, Port))
    playerID = soc.recv(1024)
    playerID = int.from_bytes(playerID, byteorder='big')
    print("You are connected to the server... ")
    print("You are player ", playerID)
    print("Preparing the game... ")
    """
    Preparing the game
    """
    # --- Create the window

    # Initialize Pygame
    pygame.init()

    # Setup the screen
    display = Display.Display("client")
    screen = display.getScreen()

    # --- Sprite lists
    # List of each player
    playerList = SpritesClass.PlayerList()

    # List of each bullet
    bulletList = SpritesClass.BulletList()

    # --- Create the sprites
    if playerID == 1:
        player = SpritesClass.Player(50,250,270,1)
        enemy = SpritesClass.Player(450,250,90,2)
    else:
        enemy = SpritesClass.Player(50,250,270,1)
        player = SpritesClass.Player(450,250,90,2)
    playerList.append(player)
    playerList.append(enemy)

    # Loop until the user clicks the close button.
    done = False
    end = False

    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()

    # Send a signal to tell the server that you are ready and
    # wait for the server to start the game
    print("Waiting for the server to start the game")
    soc.send((5).to_bytes(1, byteorder='big'))
    ready = soc.recv(1024)
    print("Game Starting...")
    # -------- Main Program Loop -----------
    while not done:
        # --- Input Processing
        playerAct, status = InputHandling.checkInput()
        playerAct[0] = player.getId() - 1
        actInt = Functions.binListToInt(playerAct)

        # Sending the player's act input list to the server
        soc.send(actInt.to_bytes(1, byteorder='big'))

        # Wait for the server to send the enemy's act input list
        enemyAct = soc.recv(1024)
        enemyAct = int.from_bytes(enemyAct, byteorder='big')
        enemyAct = Functions.intToBinList(enemyAct, 8)

        # --- Game logic

        # Create bullet if the player/enemy shoots
        if playerAct[7] == 1:
            bulletList.append(SpritesClass.Bullet(player))
        if enemyAct[7] == 1:
            bulletList.append(SpritesClass.Bullet(enemy))

        # Call the update() method on all the existing sprites
        player.update(playerAct)
        enemy.update(enemyAct)
        bulletList.updateAll()

        # --- Collision Processing
        if bulletList.getList() != 0:
            for bullet1 in bulletList.getList():
                if playerList.getList() != 0:
                    for player1 in playerList.getList():
                        dx = abs(bullet1.getPosition()[0] - player1.getPosition()[0])
                        dy = abs(bullet1.getPosition()[1] - player1.getPosition()[1])
                        if dx**2 + dy**2 < (bullet1.getRadius() + player1.getRadius())**2:
                            print("Player ", player1.getId(), " got Hit!")
                            bulletList.remove(bullet1)
                            player1.getHit()
                            if player1.isDead():
                                done = True
                                end = True
                                winner = bullet1.getShooter().getId()

                for bullet2 in bulletList.getList():
                    if bullet2 != bullet1:
                        dx = abs(bullet1.getPosition()[0] - bullet2.getPosition()[0])
                        dy = abs(bullet1.getPosition()[1] - bullet2.getPosition()[1])
                        if dx**2 + dy**2 < (bullet1.getRadius() + bullet2.getRadius())**2:
                            bulletList.remove(bullet1)
                            bulletList.remove(bullet2)
        if playerList.getList() != 0:
            for player1 in playerList.getList():
                for player2 in playerList.getList():
                    if player1 != player2:
                        dx = abs(player1.getPosition()[0] - player2.getPosition()[0])
                        dy = abs(player1.getPosition()[1] - player2.getPosition()[1])
                        if dx**2 + dy**2 < (player1.getRadius() + player2.getRadius())**2:
                            player1.cancelUpdate()
                            player2.cancelUpdate()

        # Update the screen
        screen.fill((255,255,255))
        display.drawAllPlayers(playerList)
        display.drawAllBullets(bulletList)
        display.maskFOV(player)
        display.drawLaserSight(player)
        display.drawPlayer(player)

        # Go ahead and update the screen with what is drawn.
        pygame.display.flip()

        # --- Limit to 20 frames per second
        clock.tick(60)

    if end == True:
        if winner == playerID:
            print("Congratulations! You win this match.")
        else:
            print("Sorry, your enemy wins this match.")
    pygame.quit()
