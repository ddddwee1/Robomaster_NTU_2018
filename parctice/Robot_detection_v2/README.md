# Board detection

This folder is created for first practice for Robomasters 2018.

## Methods

The two detection approaches will be based on RPN(region proposal network) and a detection net, and end-to-end network structure, inspired from [Faster-CNN](https://arxiv.org/abs/1506.01497) and [SSD](https://arxiv.org/abs/1512.02325).

### Structure 1 

The whole network contains a convolution head, RPN body and a classification back. Since the detection system will be implemented on the mobile device with low computational power, we simplify the model structure and therefore reduce the computing time. 

### Structure 2 

The network 2 is an end-to-end network that consists of multi-scale detection layers, and the model is trained with data-augmentation and hard negative mining. 