# caffe2nanoai User Guide (v0.5)

## Introduction

caffe2nanoai is an ubuntu utility to convert caffe model to nanoai model for futher inference procesures.

## Dependencies

- [Protobuf 2.6.1](https://github.com/protocolbuffers/protobuf/tree/2.6.1-artifacts)
- [Opencv-3.4.10](https://github.com/opencv/opencv/tree/3.4.10)

## Usage

caffe2nanoai supports both float inference and int8 quantized inference models. 

For float inferences, only a caffe model is needed as input.

For the int8 quantized inferences, other than the input caffe model, a calibration dataset is required. The calibration dataset will impact the accuracy of the int8 inference result, so it is highly recommended to use the training and validation dataset for calibration.

### float model generation

```shell
> ./caffe2nanoai [caffe_proto] [caffe_model] [name] [output_folder]
```

`caffe_proto`: .prototxt file of the caffe model.

`caffe_model`: .caffemodel file of the caffe model.

`name`:  name of the nanoai model.

`output_folder`: output folder of the generated nanoai model file.

For example, to convert lenet caffe model to nanoai lenet model file:

```shell
> ./caffe2nanoai lenet_m.prototxt lenet_m.caffemodel lenet ./nanoai/lenet
```

The nanoai float model file can be found at `./nanoai/lenet/lenet_nanoai.h`.

### int8 model generation

```shell
> ./caffe2nanoai [caffe_proto] [caffe_model] [name] [means[3]] [norms[3]] [images] [output_folder]
```

`caffe_proto`: .prototxt file of the caffe model.

`caffe_model`: .caffemodel file of the caffe model.

`name`: name of the nanoai model.

`means`: mean value for R, G, B channels, usually it is 127.5f. It should be aligned with the mean of your training.

`norms`: normalization value for R, G, B channels, usually it is 0.0078125. It should be aligned with the normalization of your training.

`images`: calibration image dataset folder. It is recommended to align the image resolution with your model, else the images will be resized during quantization.

`output_folder`: output folder for the generated nanoai model file.

For example, to convert lenet caffe model to nanoai int8 lenet model file:

```shell
> ./caffe2nanoai lenet_m.prototxt lenet_m.caffemodel lenet 0 0 0 1.0 1.0 1.0 mnist_dataset ./nanoai/lenet
```
***Note***: we are using the mean value 0 and norm value 1.0, that means we won't do the mean and normalization.

The nanoai int8 model file can be found at `./nanoai/lenet/lenet_int8_nanoai.h`
