# caffe2nanoai User Guide (v0.5)

## Introduction

caffe2nanoai is an ubuntu utility to convert caffe model to nanoai model for inferencing with the nanoai inference engine.

## Dependency
Below 2 extra components are needed for the caffe2nanoai utility:

- [Protobuf 2.6.1](https://github.com/protocolbuffers/protobuf/tree/2.6.1-artifacts)
- [Opencv-3.4.10](https://github.com/opencv/opencv/tree/3.4.10)

## Usage

caffe2nanoai supports the float inference and the int8 quantized inference model. 

For the float inference, only the caffe model is needed.

For the int8 quantized inference, except the caffe model, a calibration dataset is also needed. As the calibration dataset will impact the accuracy of the int8 inference result, the training and the validation dataset is highly recommended for the int8 nanoai model generation.

### float model generation

./caffe2nanoai [caffe_proto] [caffe_model] [name] [output_folder]

```
caffe_proto: the .prototxt file of the caffe model.

caffe_model: the .caffemodel file of the caffe model.

name:  the name of the nanoai model.

output_folder: the output folder of the generated nanoai model file.
```

For example, convert lenet caffe model to nanoai lenet model file
```
./caffe2nanoai lenet_m.prototxt lenet_m.caffemodel lenet ./nanoai/lenet
```

The nanoai float model file will be generated under ./nanoai/lenet/lenet_nanoai.h

### int8 model generation

./caffe2nanoai [caffe_proto] [caffe_model] [name] [means[3]] [norms[3]] [images] [output_folder]
```
caffe_proto: the .prototxt file of the caffe model.

caffe_model: the .caffemodel file of the caffe model.

name: the name of the nanoai model.

means: the mean value for R, G, B channels, usually it is 127.5f. It should be aligned with the mean of your training.

norms: the normalization value for R, G, B channels, usually it is 0.0078125. It should be aligned with the normalization of your training.

images: the calibration image dataset folder, it's better to align the image resolution with your model, else the resize will be added during the quantization.

output_folder: the output folder of the generated nanoai model file.
```

For example, convert lenet caffe model to nanoai int8 lenet model file:
Note: we are using the mean value 0 and norm value 1.0, that means we won't do the mean and normalization.

```
./caffe2nanoai lenet_m.prototxt lenet_m.caffemodel lenet 0 0 0 1.0 1.0 1.0 mnist_dataset ./nanoai/lenet
```

The nanoai int8 model file will be generated under ./nanoai/lenet/lenet_int8_nanoai.h
