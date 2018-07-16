# Board detection

This folder is created for first practice for Robomasters 2018.

## Methods

The detection approach will be based on RPN(region proposal network) and a verification net, inspired by [Faster-CNN](https://arxiv.org/abs/1506.01497).

### Structure

The whole network contains a convolution head, RPN body and a classification back. Since the detection system will be implemented on the mobile device with low computational power, we simplify the model structure and therefore reduce the computing consumption.


### Annotation sample

The sample annotation file is provided in this directory. 

The 4 points are (right_btm_x, right_btm_y, width, height). Each component is separated by tab (\t).

### In addition

We use different structure for sentry base cameras. Limited by USB bandwidth, the base cameras are only able to receive small images. Therefore an efficient network structure is implemented for sentry. 

Check 'model_zoo/07_16/netpart_s' for detail.