# Armour plate detection

This folder is created for first practice for Robomasters 2018.

## Method

The detection approach is based on MSRPN(multi-scale region proposal network) and a verification net, inspired by [Faster-RCNN](https://arxiv.org/abs/1506.01497). Actually you don't need to know Faster-RCNN because I just want to use a reference to make the readme looks good. Our approach is quite different from faster-rcnn.

## Structure

The whole network contains a MSRPN head and a classification back. Since the detection system will be implemented on the mobile device with low computational power, we simplify the model structure and therefore reduce the computing consumption.

### Multi-scale region proposal network

The feature is simple. But if you have some knowledge about convolution, it's obvious that single output layer is not enough to detect the pattern in variant scales. Naturally, we increase the network to 3 different scales.

### Verification network

After proposing the regions, we choose the top-n regions from each level, crop them, resize them, classify them with a simple network, do a non-max supression, finally output them. It can be done within 100 lines of codes.

## Annotation sample

The sample annotation file is provided in this directory. 

The 4 points are (right_btm_x, right_btm_y, width, height). Each component is separated by tab (\t).

## In addition

We use a different structure for sentry base cameras. Limited by USB bandwidth, the base cameras are only able to receive small images. Therefore an efficient network structure is implemented for sentry. 

Check 'rpn_s' and 'veri_s' for detail.

I can't tell the method about how to determine the network structure. I just feel more comfortable to build network like that.
