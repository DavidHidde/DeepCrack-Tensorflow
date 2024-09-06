# DeepCrack-Tensorflow

This is a simple Tensorflow 2+ implementation of the [DeepCrack](http://mvr.whu.edu.cn/pubs/2019-tip-deepcrack.pdf) segmentation model.  
This is a Tensorflow translation of the [original PyTorch implementation](https://github.com/qinnzou/DeepCrack) using the TensorFlow functional API. The implementation has been verified using TensorFlow 2.16.  

There are some minor differences between the PyTorch code, the paper and this implementation. Compared to the PyTorch implementation, batch normalization layers have been added after each convolution since this was indicated in the paper and SegNet. Compared to the paper, this implementation and the original PyTorch implementation implement skip layers using a 64-filter convolution, 2D upsampling and then a 1-filter convolution. From empirical testing, this has shown to improve the loss minimization of the model.

## Disclaimer

The code in this repository is based on:  

* [danielenricocahall/Keras-SegNet](https://github.com/danielenricocahall/Keras-SegNet) (main layer structure)
* [hanshenChen/crack-detection](https://github.com/hanshenChen/crack-detection) (reference of previous implementation)
* [qinnzou/DeepCrack](https://github.com/qinnzou/DeepCrack) (skip layers, filter values and general structure)

For older Tensorflow versions, [hanshenChen/crack-detection](https://github.com/hanshenChen/crack-detection) is recommended.
