# Caffe Model Pre-Process

## Introduction

We recommend following the steps below to pre-process caffe models:

## Upgrade the caffe model
Take the lenet caffe model as example:

```shell
> upgrade_net_proto_text lenet.prototxt lenet.prototxt
upgrade_net_proto_binary lenet.caffemodel lenet.caffemodel
```
Note: You can find upgrade_net_proto_text and upgrade_net_proto_binary from the caffe release.

## Format Conversion

We recommend to use this open source tool (https://github.com/zds79/CaffeBatchnormFuse) to format your model as follow. 

```shell
> python CaffeBatchnormFuse.py --proto=lenet.prototxt --model=lenet.caffemodel
```
This generates lenet_m.prototxt and lenet_m.caffemodel, which you can use as nanoai converter inputs.
