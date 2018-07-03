# Board detection

This folder is created for first practice for Robomasters 2018.

## Methods

The detection approache will be based on RPN(region proposal network) and a verification net, inspired by [Faster-CNN](https://arxiv.org/abs/1506.01497).

### Structure

The whole network contains a convolution head, RPN body and a classification back. Since the detection system will be implemented on the mobile device with low computational power, we simplify the model structure and therefore reduce the computing consumption.