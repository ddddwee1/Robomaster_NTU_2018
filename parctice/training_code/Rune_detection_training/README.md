# Board detection

This folder is created for first practice for Robomasters 2018.

## Methods

We split different recognition tasks into different network. Each network is light in structure.

### Structure

conv(5,16)
maxpool(2)
conv(5,16)
maxpool(2)
conv(5,16)
maxpool(2)
fc(50)
fc(10)

### Board display

In folder DigitDetection, the script utilizes pygame and display the simulated rune board.

### In addition

Testing code can be found in './practice/New_protocol/rm_sys_infantry'
