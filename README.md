# IdenNet: Identity-Aware Facial Action Unit Detection


## Introduction

This repository contains the codes for IdenNet which aims at combining AU and ID labels to train CNNs for identity-aware facial AU detection. We provide the main parts of our codes in training, testing and constructing networks. This implementation is based on [CAFFE](https://github.com/BVLC/caffe). 

## Structure 

We organize our codes as following: 

* `caffe/` should contain the codes for caffe framework. 
* `train.py`, `test.py`, `construct_network_protos.py` and `construct_solver_protos.py` are used for training, testing and constructing prototxts as specified by their names. 
* `utils/` contains the codes for describing the architectures of IdenNet and other helper functions.
	* `utils/layers/data_layer.py` contains the data layer that generates triplets. 
	* `utils/netowrk/IdenNet.py` specifies the IdenNet architecture. 
	* `utils/metric.py` contains the codes for calculating f1 score. 

According to the caffe framework, please construct the network and solver prototsts first. Then, CNNs can be trained by the caffe's tools or the provided codes. The [BP4D](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html) and the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) datasets are utilized for training IdenNet. Please refer to the corresponding websites for instructions to acquire the data. 


## Contact 

Please send emails to andytu455176 at gmail.com if you have any questions about the codes. 
