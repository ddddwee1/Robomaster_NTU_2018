# Robomasters Software Team

This repo is created for Robomasters 2018. Software Team of MECATron, [Nanyang Technological University](http://www.ntu.edu.sg).

## Robomaster robot system

I think it's also necessary to write some description for this.

### Write something before

This code is NOT the final version which we used in Robomasters 2018. We made some minor modifications and I could not find the files.

This repo is an extension for ICRA robot system.

Now the code can be finished in one day.

We used python2.7 and tensorflow 1.7.0 for testing.

### Code structure

The code is structured as following:

```
main
|- data retriever
|  |- control mod (control_test.py)
|
|- camera module
|
|- armour plate mod
|  |- detection mod
|  |  |- net
|  |  |  |- modelveri_tiny (for big images)
|  |  |  |- modelveri_tiny_s (for small images/ only for sentry)
|
|- rune mod
|  |- digit detection
|  |  |- conv_deploy
|  |
|  |- rune_shooting_logic
|
|- robot prop
|
|- util

```

### Main

As name suggests, the main.py is the entry point of this program. It creates a data_reader object to continuously communicate with MCU and a camera thread to read images from camera.

### Data retriever

Read properties from MCU and update robot_prop.

### camera module

See the code, simple. Create camera object on creation, read images after run.

### armor_plate_mod

Call detection_mod and do auto-shooting.

### detection_mod

Detection module for armour plate. 

Pipeline:

Multi-scale region proposal network -> crop top-n from each scale -> verificate proposals -> non-max supression.

### rune_mod

Call rune_module to detect the different runes.

### rune_module

Contains model for 7-segment, handwritten and flaming digits. The codes are quite long but very easy to understand.

### Robot_prop

Robot properties. Stores some variables like turret speed, robot coordinates, robot angle, etc.

### util

Utilities. Basically some converting functions will be put here.

### Write after

If someone is really reading this message here, I would tell you that the recognition and auto-shooting is useful only when you have very strong hardware and MCU fondamentals.

Otherwise.

If you have anything to discuss with me, can Email me: chengyu996@gmail.com

### Acknowledgement

Thanks to everyone in software team for labelling the training data.

Thanks to Guowei and Hansel for testing testing and testing. 

Thanks to Guowei for coding the rune module.

Thanks to Hansel for auto-shoot.

Thanks to Vinsens for writing the rune board simulator and first version of board detection.

Thanks to Fang Meiyi for hardware and environment setup.

Thanks to myself for writing other codes. 
