import pygame
from pygame.locals import *
import random, math, sys
import Display
import Functions
import SpritesClass
import InputHandling
from socket import *
import time, random
import datetime

# Setting up the port
Port, HostIP = 27400, ""

Port = int(input("Please input the port for this game : "))

with socket(AF_INET, SOCK_STREAM) as soc:
    soc.bind((HostIP, Port))
    soc.listen(100)
    print("The server is now running and accepting 2 players!")

    # Waiting for connection from player 1
    conn1, addr1 = soc.accept()
    print("Player 1 Connected. Waiting for player 2...")
    conn1.send((1).to_bytes(1, byteorder='big'))

    # Waiting for connection from player 2
    conn2, addr2 = soc.accept()
    print("Player 2 Connected. Preparing the game...")
    conn2.send((2).to_bytes(1, byteorder='big'))
    """
    Prepare the log file to store the actions
    """
    fName = datetime.datetime(2017,12,12)
    fName = fName.now()
    fName = fName.strftime("%Y-%m-%d-%H-%M-%S")
    fNameP1 = "./GameLog/" + fName + "-P1.csv"
    fNameP2 = "./GameLog/" + fName + "-P2.csv"
    with open(fNameP1, 'a') as fP1:
        with open(fNameP2, 'a') as fP2:
            """
            Prepare the simulation of the game in the server
            """
            # --- Create the window

            # Initialize Pygame
            pygame.init()

            # Setup the screen
            display = Display.Display("server")
            screen = display.getScreen()

            # --- Sprite lists
            # List of each player
            playerList = SpritesClass.PlayerList()

            # List of each bullet
            bulletList = SpritesClass.BulletList()

            # --- Create the sprites
            player1 = SpritesClass.Player(50,250,270,1)
            player2 = SpritesClass.Player(450,250,90,2)
            playerList.append(player1)
            playerList.append(player2)

            # Loop until the user clicks the close button.
            done = False
            end = False

            # Used to manage how fast the screen updates
            clock = pygame.time.Clock()

            # Wait for the ready signals from the two clients
            readyP1 = conn1.recv(1024)
            readyP2 = conn2.recv(1024)
            print("Both players are ready... ")
            print("Commencing the game!")
            conn1.send((5).to_bytes(1, byteorder='big'))
            conn2.send((5).to_bytes(1, byteorder='big'))

            # -------- Main Program Loop -----------
            while not done:
                # Calculating the enemy's relative location and log it
                log1 = Functions.logVisibleEnemyPosition(playerList.getList()[0], playerList.getList()[1])
                log2 = Functions.logVisibleEnemyPosition(playerList.getList()[1], playerList.getList()[0])

                # Reward for training AI purposes
                reward = [0.05,0.05]

                # Waiting for input from the two clients
                actP1 = conn1.recv(1024)
                actP2 = conn2.recv(1024)

                # Sending the input to the other player
                conn1.send(actP2)
                conn2.send(actP1)

                # Processing the input
                actP1 = int.from_bytes(actP1, byteorder='big')
                actP2 = int.from_bytes(actP2, byteorder='big')
                actP1 = Functions.intToBinList(actP1, 8)
                actP2 = Functions.intToBinList(actP2, 8)

                # Logging the input
                log1 = log1 + Functions.logActionTuple(actP1)
                log2 = log2 + Functions.logActionTuple(actP2)

                # --- Game logic

                # Create bullet if the player1/player2 shoots
                if actP1[7] == 1:
                    bulletList.append(SpritesClass.Bullet(playerList.getList()[0]))
                    reward[0] -= 0.1

                if actP2[7] == 1:
                    bulletList.append(SpritesClass.Bullet(playerList.getList()[1]))
                    reward[1] -= 0.1

                # Call the update() method on all the existing sprites
                playerList.getList()[0].update(actP1)
                playerList.getList()[1].update(actP2)
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
                                    reward[player1.getId() - 1] -= 1.05
                                    reward[bullet1.getShooter().getId() - 1] += 1

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

                # Go ahead and update the screen with what is drawn.
                pygame.display.flip()

                # Convert the reward to string and add to the respective log
                log1 = log1 + str(reward[0]) + "\n"
                log2 = log2 + str(reward[1]) + "\n"

                # Write the log to the csv file
                fP1.write(log1)
                fP2.write(log2)

                # --- Limit to 20 frames per second
                clock.tick(60)

            if end == True:
                print("Game has ended.")
                print("Player ", winner, " Wins.")
            pygame.quit()
