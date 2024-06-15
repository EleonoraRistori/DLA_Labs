# Lab 1 CNN
This first laboratory is dedicated to carrying out some 
experiments on convolutional neural networks.
This is the only laboratory implemented using Jupiter Notebooks.
The results and considerations can be found inside each notebook.

## Contents
* **Exercise 1**:
We trained a simple Multilayer Perceptron (MLP), a CNN and a RNN on the 
CIFAR10 dataset. We monitored both the loss and accuracy
on the train and validation sets. Code is available in `Exercise1.ipynb`.

* **Exercise 2.2**: 
We Fully-convolutionalized a traditional convolutional neural network. 
First we trained a traditional CNN on MNIST than we turn it into a 
network that can predict classification outputs at all pixels in an 
input image. We evaluate the results by creating new datasets that position the MNIST
digits in random positions in a larger image. Code is available in `Exercise2_2.ipynb`.

* **Exercise 2.3**: 
Here we have explained the performance of a CNN trained on CIFAR10 implementing
[Grad-CAM](https://arxiv.org/abs/1610.02391), which returns an activation 
map showing where the network focuses most to carry out the classification.
Code is available in `Exercise2_3.ipynb`.