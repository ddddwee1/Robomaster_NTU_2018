# Robomasters Software Team

This repo is created for Robomasters 2018. Software Team of MECATron, [Nanyang Technological University](http://www.ntu.edu.sg).

## ICRA robot system

I think it's necessary to write some description for this.

### Write something before

This code is NOT the final version which we used in ICRA 2018. Here the code does not include any strategy and obstacle avoid stuff. I cannot find the final code but I think it will be easy to insert those modules into this system. 

The purpose of pushing this code is just to illustrate how easy it will be for implementing a control system if we have a good MCU programmer. 

It's really simple that the code can be finished within two days. 

Thanks to e-team!

Okay, but I have to say that we spent days and days for debugging.

We used python2.7 and tensorflow 1.7.0 for testing.

### Code structure

The code is structured as following:

```
main
|- data retriever
|  |- control mod
|
|- motion main
|  |- mp_util
|  |  |- control template
|  |  |- icramap
|  |  |  |- Astar
|
|- detection main
|  |- camera module
|  |- PID 2D
|  |  |- PID
|  |- detection mod
|  |  |- net
|
|- robot prop
|
|- util
|
|- network (not in use)
|
|- lidar (not in use)
```

### Main

As name suggests, the main.py is the entry point of this program. It creates a data_reader object to continuously communicate with MCU.

Then it will continuously do motion planning and armour plate detection.

### Data retriever

Read properties from MCU, including coordinates (w,h) and current robot angle.

### Control mod

Control module. In charge of communicating with MCU. It encodes and decodes the information.

### Motion main

Entry point for motion planning. High level wrap-up for battleground map, motion planning and velocity computation.

### mp_util.control_template

Convert the (robot angle, point_start, point_end) to (left_right_velocity, forward_backward_velocity).

### mp_util.icramap

Build icramap and call astar algorithm for path planning.

### mp_util.Astar

Simple Astar implementation. Don't need to explain, you can find tons of explanation on [Google](www.google.com) or [baidu](www.baidu.com) or [bing](www.bing.com).

### detection main

Entry point for armour plate detection. The object will create a camera thread to continuously read images from camera, and then utilizes detection module and turret speed conversion.

### camera module

See the code, simple. Create camera object on creation, read images after run.

### detection_mod

Detection module. 

Pipeline:

Multi-scale region proposal network -> crop top-n from each scale -> verificate proposals -> non-max supression.

### PID

I dont want to talk about this.

### Robot_prop

Robot properties. Stores some variables like turret speed, robot coordinates, robot angle, etc.

### util

Utilities. Basically some converting functions will be put here.

### network

Network scripts. Including server and client. The scripts are not tested yet, but it will basically look like that.

### lidar

Lidar functions. The communication is based on RP-LIDAR-V2 and lidar module has been tested already. However, we found that using uwb is more efficient than lidar then we delete the lidar module in system main entry.

The fit function basically samples the battleground every 5cm/9deg and compute their L1 error (absolute error). Then choose the most confident point. 

### Write after

I thought to use ROS for robot development once before, but write the codes from plain will make more sense. 

The functionality can be coded in a very simple manner. As you can see, the whole system can be programmed within 1000 lines of code. 

I heard from other teams that their code are more than 5k lines. That still confuses me till now. I have no idea because no complex functionalities are observed in competition. I wonder if they are making more efforts on stability control for robot.

### Acknowledgement

Thanks to everyone in software team for labelling the training data.

Thanks to Fang Meiyi for hardware and environment setup, PID adjustment, turret testing. She also did lots of labelling work.

Thanks to Guo Tai for implementing Astar algorithm.

Thanks to myself for writing other codes. 

Thanks to e-team for wiring and building MCU. They provided really reliable programs to us.
