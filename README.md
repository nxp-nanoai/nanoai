# ![](./doc/NanoAI_MCU.PNG) Nano.AI
**N**ano.AI is a lightweight neural network inference framework which is fully optimized for NXP 's MCU(arm m-core) platform. 

It took into consideration the limitation of memory and computing resources of MCU platforms from design to implementation.

## Features

- Pure c implementation without 3rd-party library dependency
- Support for float and int8 quantized inference
- Efficient memory reuse
- Full optimization in performance and accuracy for MCU int8 quantized inference
- Easy deployment onto MCU platform

---

## Architecture

![](./doc/NanoAI_Architecture.PNG)

**N**ano.AI can be divided into a ***converter*** and an ***Inference Engine***.  

#### **Converter**

**N**ano.AI converter prepares the pre-trained models for further deployment. Currently it only supports Caffe models as input (more popular models like ONNX/Tensorflow/Pytorch/MXNET will be supported in the future). You can use open source tools to re-format your model to Caffe format before feeding it into Nano.AI converter.

- The convert also includes a quantization tool to generate quantization information for int8 quantized inferences. Currently the quantization tool supports Caffe format models only. 

#### **Inference Engine**

- The inference engine constructs the network graph from Nano.AI format inputs and returns the outputs to the caller.
- It does not allocate any memory directly. It requests buffer through APIs and let the caller allocate buffers for it. The whole inference process uses caller-provided buffers only.
- Currently Nano.AI supports float and int8 quantized inferences. Int8 inference process is fully optimized with NXP MCU int8 accelerations, while float inference is not fully optimized and is only for accuracy comparison purpose. Typically float graphs uses 4 times memory compared with int8 graphs but have in a higher accuracy. Customers can decide which inference method to used considering their available memory and performance and accuracy requirements.

****

## Deployment
![](./doc/NanoAI_Deployment.PNG)

**N**ano.AI typically includes ***Pre-Process***, ***Convert***, and ***Inference*** stages in the deployment: 

#### **Pre-Process Stage**    [[user guide](./doc/preprocess_user_guide.md)]

At this stage we can use open source tools for offline model formatting, for example:
     - BatchNorm and Scale Merge to Previous Convolution
     - Other operation fusion

#### **Convert stage**    [[user guide](./doc/tools_user_guide.md)]

At convert stage input caffe models are converted into Nano.AI defined models. For int8 quantized models, a calibration dataset is required to produce quantization information. The calibration dataset can affect the accuracy of int8 inference result so it is highly recommended to use the training, validation and testing data of the input model as the calibration dataset.

#### **Inference stage**    [[engine API](./doc/engine_api.md)]

At inference stage the engine takes the Nano.AI format model from converter, with some additional input data, and carry out graph inference. It then extracts the output data for the caller. - Popular network such as LENET, CIFAR10 are privided as SDK deployment examples for users' reference.

****

## Nano.AI model definition

A Nano.AI model is represented as an ***array*** in C. It has 3 components as described below:

#### **Blob ID Definition**

Each blob has a unique ID. A blob can be input, output or intermedia.

```c
enum {
    LENET_INT8_B_DATA_ID = 0,
    LENET_INT8_B_CONV1_ID = 1,
    ...
    LENET_INT8_B_PROB_ID = 8,
    LENET_INT8_B_COUNT
};
```
#### **Model**

The model graph information:

A layer includes:

- Layer information: type, parameters
- Bottom blobs
- Top blobs

```c
NANONN_MODEL static NANONN_CONST unsigned char BINARY_MODEL_ALIGN lenet_int8_binary_model[] = {
0x5a,0x44,0x00,0x01,0x09,0x00,0x09,0x00,0x02,0x00,0x01,0x00,0x09,0x00,0x02,0x00,
0x18,0x02,0x00,0x00,0x24,0x00,0x00,0x00,0x30,0x00,0x00,0x00,0xae,0x00,0x00,0x00,
...
0x03,0x9b,0x39,0x3d,0x06,0x00,0x01,0x00,0xf0,0x08,0x00,0x00,0x1e,0xbd,0x0c,0x3d,
0x07,0x00,0x0a,0x00,0xf4,0x08,0x00,0x00,0xa7,0xf0,0x28,0x3d,0x08,0x00,0x01,0x00,
};
```
#### **Model Data**

Weights/bias for each layer includes:

- Weights/Bias information: size, format
- Weights/Bias data

```c
NANONN_MODEL_DATA static NANONN_CONST unsigned char BINARY_MODEL_ALIGN lenet_int8_binary_model_data[] = {
0x64,0x00,0x00,0x00,0x58,0x02,0x00,0x00,0x03,0x00,0x02,0x00,0xa8,0x61,0x00,0x00,
0x32,0x00,0x00,0x00,0xa8,0x02,0x00,0x00,0x50,0x64,0x00,0x00,0x05,0x00,0x02,0x00,
...
0xbf,0xa8,0x17,0x3d,0x41,0x16,0x30,0x3d,0xbd,0x2a,0x31,0x3d,0xc6,0x08,0x35,0x3d,
0x9e,0x63,0x35,0x3d,0x8c,0x1c,0x27,0x3d,0xe2,0xc7,0x20,0x3d,0x3c,0xbf,0x27,0x3d,
};
```

---

## Supported Operators

| Index | Operator             | Comments | Reference |
| ----- | -------------------- | -------- | --------- |
| 0     | Input                |          |           |
| 1     | Convolution          |          |           |
| 2     | DepthwiseConvolution |          |           |
| 3     | Pooling              |          |           |
| 4     | Relu                 |          |           |
| 5     | Prelu                |          |           |
| 6     | Innerproduct         |          |           |
| 7     | Split                |          |           |
| 8     | Scale                |          |           |
| 9     | Eltwise              |          |           |
| 10    | Softmax              |          |           |
| 11    | Dropout              |          |           |
| 12    | Concat               |          |           |
## Benchmark

| Model   | Model Size(Kbytes) | Input   Dimension(C x H x W) | Runtime Memory(Kbytes) | [Single Inference Time on NXP RT106F  Kit(ms)](https://www.nxp.com/design/designs/nxp-edgeready-mcu-based-solution-for-face-recognition:MCU-FACE-RECOGNITION) | MSE With Float | Similarity With Float | Calibration  Data set           |
| ------- | ------------------ | ---------------------------- | ---------------------- | ------------------------------------------------------------ | -------------- | --------------------- | ------------------------------- |
| mnist   | 426                | 1x28x28                      | 18                     | 9.5                                                          | 0.000063       | 1.0                   | mnist train and test data set   |
| cifar10 | 90                 | 3x32x32                      | 45                     | 38.5                                                         | 0.009022       | 0.999872              | cifar10 train and test data set |
|         |                    |                              |                        |                                                              |                |                       |                                 |


---

## Community
WeChat Group "Nano.AI"

![](./doc/NanoAI_WeChat.PNG)
---

## References

**N**ano.AI refers to the following famous open source projects:
- [Caffe](https://github.com/BVLC/caffe)
- [Protobuffer](https://github.com/protocolbuffers/protobuf)
- [NCNN](https://github.com/Tencent/ncnn)
- [ONNX](https://github.com/onnx/onnx)
- [MNN](https://github.com/alibaba/MNN)
- [CMSIS NN](https://arm-software.github.io/CMSIS_5/NN/html/index.html)
- [Open CV](https://opencv.org/)

---
Copyright 2016-2020 NXP