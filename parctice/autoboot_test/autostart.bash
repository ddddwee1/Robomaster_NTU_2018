#!/bin/bash
#!/usr/bin/env python
# Author: Guo Wei

echo -e "\e[1;33m >>> Beginning to run jetson"

	cd /home/nvidia
	echo nvidia | sudo -S ./jetson_clocks.sh

echo -e "\e[34m >>> Run python script main_combine"

	cd /home/nvidia/Desktop/rm_sys_infantry
	chmod +x main_combine.py
	python ./main_combine.py
