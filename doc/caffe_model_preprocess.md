# Caffe Model Pre-Process

## Introduction

We recommend follow below steps to do the pre-process for the caffe model:

## Upgrade the caffe model
Take the lenet caffe model as example:

```
upgrade_net_proto_text lenet.prototxt lenet.prototxt
upgrade_net_proto_binary lenet.caffemodel lenet.caffemodel
```
Note: You can find the upgrade_net_proto_text and upgrade_net_proto_binary from the caffe release.

## Do the pre-process

We recommend to use this open source tool (https://github.com/zds79/CaffeBatchnormFuse) to do the pre-process. 

Take the lenet caffe model as example:

```
python CaffeBatchnormFuse.py --proto=lenet.prototxt --model=lenet.caffemodel
```
The pre-processed lenet_m.prototxt and lenet_m.caffemodel will be generated, and you can use them for the nanoai model generation in the next.
