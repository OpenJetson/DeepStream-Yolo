# DeepStream-Yolo
NVIDIA DeepStream SDK configuration for Yolo model

Tested on NVIDIA Jetson Nano

Comparison between NVIDIA DeepStream SDK and Darknet: https://github.com/marcoslucianops/Benchmark-Yolo

##

* [Requirements](#requirements)
* [Editing default model](#editing-default-model)
* [Compiling edited model](#compiling-edited-model)
* [Editing yolo.cfg file](#editing-yolocfg-file)
* [Understanding and editing deepstream_app_config](#understanding-and-editing-deepstream_app_config)
* [Understanding and editing config_infer_primary](#understanding-and-editing-config_infer_primary)
* [Testing model](#testing-model)
* [Custom functions in your model](#custom-functions-in-your-model)
* [FAQ](#faq)

##

### Requirements
* [NVIDIA DeepStream SDK 5](https://developer.nvidia.com/deepstream-sdk)
* [GStreamer](https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html)
* [Pre-treined Yolo model](https://github.com/AlexeyAB/darknet) (for NVIDIA Jetson Nano, I recommend YoloV3-Tiny or YoloV3-Tiny-PRN <- Fastest)

##

### Editing default model
1. Copy nvdsinfer_custom_impl_Yolo folder (located in /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo/) to your custom yolo directory (must be in sources folder).
2. Edit Yolo DeepStream for your custom model:

* Example for YoloV3-Tiny:

Line 34:
```
static const int NUM_CLASSES_YOLO = 80; // Number of classes of your custom model
```

Line 299-304:
```
    static const std::vector<float> kANCHORS = {
        10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319}; // Anchors of your custom model
    static const std::vector<std::vector<int>> kMASKS = {
        {3, 4, 5}, // First mask of your custom model
        {1, 2, 3}}; // Second mask of your custom model
```

3. Copy and remane your obj.names file to labels.txt to your custom yolo directory.
4. Copy your yolo.cfg (v3, v3-tiny, etc.) file to your custom yolo directory.
5. Copy config_infer_primary.txt and deepstream_app_config.txt (same of your yolo model; v3, v3-tiny, etc.) from /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo to your custom yolo directory.

##

### Compiling edited model
1. Check your CUDA version (nvcc --version)
2. Open terminal
3. Go to your custom yolo directory
4. Type this command (example for CUDA 10.2 version):
```
CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo
```

##

### Editing yolo.cfg file
Set batch=1 and subdivisions=1
```
[net]
# Testing
batch=1
subdivisions=1
# Training
#batch=64
#subdivisions=16
```

##

### Understanding and editing deepstream_app_config
To understand and edit deepstream_app_config file, read the [DeepStream SDK Development Guide - Configuration Groups](https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html#page/DeepStream_Development_Guide/deepstream_app_config.3.2.html)

In this repository have example of deepstream_app_config_yoloV3_tiny.txt file for YoloV3-Tiny.

##

* Edit tiled-display

```
[tiled-display]
enable=1
# If you have 1 stream use 1/1 (rows/columns), if you have 4 streams use 2/2 or 4/1 or 1/4 (rows/columns)
rows=1
columns=1
# Resolution of tiled display
width=1280
height=720
gpu-id=0
nvbuf-memory-type=0
```

##

* Edit source

Example for 1 source:
```
[source0]
enable=1
# 1=Camera (V4L2), 2=URI, 3=MultiURI, 4=RTSP, 5=Camera (CSI; Jetson only)
type=3
# Stream URL
uri=rtsp://192.168.1.2/Streaming/Channels/101/httppreview
# Number of sources copy (if > 1, you need edit rows/columns in tiled-display section and batch-size in streammux section and config_infer_primary_yoloV3_tiny.txt; need type=3 for more than 1 num-sources)
num-sources=1
gpu-id=0
cudadec-memtype=0
```
Example for 1 duplcated source:
```
[source0]
enable=1
type=3
uri=rtsp://192.168.1.2/Streaming/Channels/101/httppreview
num-sources=2
gpu-id=0
cudadec-memtype=0
```
Example for 2 sources:
```
[source0]
enable=1
type=3
uri=rtsp://192.168.1.2/Streaming/Channels/101/httppreview
num-sources=1
gpu-id=0
cudadec-memtype=0

[source1]
enable=1
type=3
uri=rtsp://192.168.1.3/Streaming/Channels/101/httppreview
num-sources=1
gpu-id=0
cudadec-memtype=0
```

##

* Edit sink

Example for 1 source or 1 duplicated source:
```
[sink0]
enable=1
# 1=Fakesink, 2=EGL (nveglglessink), 3=Filesink, 4=RTSP, 5=Overlay (Jetson only)
type=2
# Indicates how fast the stream is to be rendered (0=As fast as possible, 1=Synchronously)
sync=0
# The ID of the source whose buffers this sink must use
source-id=0
gpu-id=0
nvbuf-memory-type=0
```
Example for 2 sources:
```
[sink0]
enable=1
type=2
sync=0
source-id=0
gpu-id=0
nvbuf-memory-type=0

[sink1]
enable=1
type=2
sync=0
source-id=1
gpu-id=0
nvbuf-memory-type=0
```

##

* Edit streammux

Example for 1 source:
```
[streammux]
gpu-id=0
# Boolean property to inform muxer that sources are live
live-source=1
# Number of sources
batch-size=1
# Time out in usec, to wait after the first buffer is available to push the batch even if the complete batch is not formed
batched-push-timeout=40000
# Resolution of streammux
width=1920
height=1080
enable-padding=0
nvbuf-memory-type=0
```
Example for 1 duplicated source or 2 sources:
```
[streammux]
gpu-id=0
live-source=1
batch-size=2
batched-push-timeout=-1
width=1920
height=1080
enable-padding=0
nvbuf-memory-type=0
```

##

* Edit primary-gie

```
[primary-gie]
enable=1
gpu-id=0
gie-unique-id=1
nvbuf-memory-type=0
config-file=config_infer_primary_yoloV3_tiny.txt
```

* You can remove [tracker] section, if you don't use it.

##

### Understanding and editing config_infer_primary
To understand and edit config_infer_primary file, read the [NVIDIA DeepStream Plugin Manual - Gst-nvinfer File Configuration Specifications](https://docs.nvidia.com/metropolis/deepstream/plugin-manual/DeepStream%20Plugins%20Development%20Guide/deepstream_plugin_details.3.01.html#wwpID0E0WDB0HA)

In this repository have example of config_infer_primary_yoloV3_tiny.txt file for YoloV3-Tiny.

##

* Edit model-color-format accoding number of channels in yolo.cfg (1=GRAYSCALE, 3=RGB)

```
# 0=RGB, 1=BGR, 2=GRAYSCALE
model-color-format=0
```

##

* Edit model-engine-file (example for batch-size=1 and network-mode=2)

```
model-engine-file=model_b1_gpu0_fp16.engine
```

##

* Edit batch-size

```
# Number of sources
batch-size=1
```

##

* Edit network-mode

```
# 0=FP32, 1=INT8, 2=FP16
network-mode=0
```

##

* Edit num-detected-classes according number of classes in yolo.cfg

```
num-detected-classes=80
```

##

* Edit network-type

```
# 0:Detector, 1:Classifier, 2:Segmentation
network-type=0
```

##

* Add/edit interval (FPS increase if > 0)

```
# Interval of detection
interval=1
```

##

* Change threshold to pre-cluster-threshold

```
threshold=0.7
```

to

```
pre-cluster-threshold=0.7
```

##

* To get more similar inference results to Darknet, change

```
nms-iou-threshold=0.3
pre-cluster-threshold=0.7
```

to

```
# Darknet nms
nms-iou-threshold=0.45
# Darknet conf_thresh
pre-cluster-threshold=0.25
```

### Testing model
To run your custom yolo model, use this command (in your custom model directory; example for yolov3-tiny):
```
deepstream-app -c deepstream_app_config_yoloV3_tiny.txt
```

##

### Custom functions in your model

You can get metadata from deepstream in Python and C. For C, you need edit deepstream-app or deepstream-test code. For Python your need install and edit [this](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps).

You need manipulate [NvDsObjectMeta](https://docs.nvidia.com/metropolis/deepstream/dev-guide/DeepStream_Development_Guide/baggage/struct__NvDsObjectMeta.html), [NvDsFrameMeta](https://docs.nvidia.com/metropolis/deepstream/dev-guide/DeepStream_Development_Guide/baggage/struct__NvDsFrameMeta.html) and [NvOSD_RectParams](https://docs.nvidia.com/metropolis/deepstream/dev-guide/DeepStream_Development_Guide/baggage/struct__NvOSD__RectParams.html) to get label, position, etc. of bboxs.

In C deepstream-app application, your code need be in analytics_done_buf_prob function.
In C/Python deepstream-test application, your code need be in tiler_src_pad_buffer_probe function.

Python is slightly slower than C (on Jetson Nano, ~2FPS).

##

### FAQ
**Q:** Can I run custom yolo model on deepstream with non-square shape?

**A:** You can, but the accuracy will greatly decrease. If you want to test, see [this patch](https://forums.developer.nvidia.com/t/trouble-in-converting-non-square-grid-in-yolo-network-to-tensorrt-via-deepstream/107541/12).

<br>

**Q:** How to make more than 1 yolo inference?

**A:** See [MULTIPLE-INFERENCES.md](https://github.com/marcoslucianops/DeepStream-Yolo/blob/master/MULTIPLE-INFERENCES.md) in this repository.

<br>

**Q:** How to use YoloV3-Tiny-PRN? (~2FPS increase on NVIDIA Jetson Nano)

**A:** Replace nvdsinfer_custom_impl_Yolo/yolo.cpp file to my [yolo.cpp](https://github.com/marcoslucianops/DeepStream-Yolo/blob/master/examples/yolov3-tiny-prn/nvdsinfer_custom_impl_Yolo/yolo.cpp) file and change config_infer.txt file from

```
custom-network-config=yolov3-tiny.cfg
model-file=yolov3-tiny.weights
```

to

```
custom-network-config=yolov3-tiny-prn.cfg
model-file=yolov3-tiny-prn.weights
```

##

I'm not an expert in DeepStream or Yolo, but I can help in any issue or question.

Sorry for any English error, it is not my native language.
