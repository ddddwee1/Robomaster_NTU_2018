# Robomasters Software Team

This repo is created for Robomasters 2018. Software Team of MECATron, [Nanyang Technological University](http://www.ntu.edu.sg).

## Training code

### Armour plate detection

We are using MS-RPN + verification network for plate detection. We are lazy, and don't want to use traditional method and tune tune tune the parameters. Give me 3 hours, write the processing codes, build a network, establish testing functions, then network will do all things for me.

The network will learn by itself, and we just need to label around 2k images. It's year 2018 already, so the AI must be able to learn that. If it fails, I'd rather go back to 2008.

### Rune detection

This one is even simpler. MNIST for small rune, and randomly do some image processing to build training dataset for big rune and 7 segmentation digits. 
