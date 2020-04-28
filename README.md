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
* [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) >= 4
* [OpenCV](https://opencv.org/releases.html) (if you want to [populate confidence](#populate-confidence); it's built-in in latest [NVIDIA JetPack](https://developer.nvidia.com/embedded/jetpack); sudo apt-get install libopencv-dev)
* [GStreamer](https://gstreamer.freedesktop.org/)-1.0 Development package and Base Plugins Development package (sudo apt-get install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev)
* [Pre-treined Yolo model](https://github.com/AlexeyAB/darknet) (for NVIDIA Jetson Nano, I recommend YoloV3-Tiny)

##

### Editing default model
1. Copy nvdsinfer_custom_impl_Yolo folder (located in /opt/nvidia/deepstream/deepstream-4.0/sources/objectDetector_Yolo/) to your custom yolo directory.
2. Edit Yolo DeepStream for your custom model (in your custom yolo directory), following this [Application Note](https://docs.nvidia.com/metropolis/deepstream/4.0/Custom_YOLO_Model_in_the_DeepStream_YOLO_App.pdf).
3. Copy and remane your obj.names file to labels.txt to your custom yolo directory.
4. Copy your yolo.cfg (v3, v3-tiny, etc.) file to your custom yolo directory.
5. Copy config_infer_primary.txt and deepstream_app_config.txt (same of your yolo model; v3, v3-tiny, etc.) from /opt/nvidia/deepstream/deepstream-4.0/sources/objectDetector_Yolo to your custom yolo directory.

In example folder, in this repository, have config_infer_primary and deepstream_app_config example files for yolov3-tiny.

##

### Populate confidence
If you want to populate confidence in object detection, do patch below (Thanks for [chandrahasj](https://forums.developer.nvidia.com/t/nvinfer-is-not-populating-confidence-field-in-nvdsobjectmeta-ds-4-0/79319/20)). If you don't want to populate confidence, skip this step.

1. Only works with OpenCV
2. You need edit C or Python deepstream program to show the confidence, this patch only allows to get confidence (view [this post](https://forums.developer.nvidia.com/t/nvinfer-is-not-populating-confidence-field-in-nvdsobjectmeta-ds-4-0/79319/20) to see more)

In /opt/nvidia/deepstream/deepstream-4.0/sources/gst-plugins/gst-nvinfer/gstnvinfer_meta_utils.cpp
```
@@ -80,7 +80,7 @@ attach_metadata_detector (GstNvInfer * nvinfer, GstMiniObject * tensor_out_objec
     obj_meta = nvds_acquire_obj_meta_from_pool (batch_meta);
 
     obj_meta->unique_component_id = nvinfer->unique_id;
-    obj_meta->confidence = 0.0;
+    obj_meta->confidence = obj.detectionConfidence;
 
     /* This is an untracked object. Set tracking_id to -1. */
     obj_meta->object_id = UNTRACKED_OBJECT_ID;
```
In /opt/nvidia/deepstream/deepstream-4.0/sources/includes/nvdsinfer_context.h
```
@@ -369,6 +369,8 @@ typedef struct
     int classIndex;
     /* String label for the detected object. */
     char *label;
+    /* detection confidence of the object */
+    float detectionConfidence;
 } NvDsInferObject;
```
In /opt/nvidia/deepstream/deepstream-4.0/sources/libs/nvdsinfer/nvdsinfer_context_impl_output_parsing.cpp
```
@@ -282,6 +282,7 @@ NvDsInferContextImpl::clusterAndFillDetectionOutputDBSCAN(NvDsInferDetectionOutp
             object.label = nullptr;
             if (c < m_Labels.size() && m_Labels[c].size() > 0)
                 object.label = strdup(m_Labels[c][0].c_str());
+            object.detectionConfidence = m_PerClassObjectList[c][i].detectionConfidence;
             output.numObjects++;
         }
     }
```
If you using libopencv-dev (opencv4), edit /opt/nvidia/deepstream/deepstream-4.0/sources/libs/nvdsinfer/Makefile (lines 29-30) 
```
CFLAGS+= -fPIC -std=c++11 \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
-	 -I ../../includes 
+	 -I ../../includes \
+	 -I /usr/include/opencv4

LIBS := -shared -Wl,-no-undefined \
```
Compile nvdsinfer (requires libopencv-dev; example for CUDA 10.0 version)
```
cd /opt/nvidia/deepstream/deepstream-4.0/sources/libs/nvdsinfer
CUDA_VER=10.0 make and sudo CUDA_VER=10.0 make install
```
Compile gst-nvinfer (requires libgstreamer-plugins-base1.0-dev and libgstreamer1.0-dev; example for CUDA 10.0 version)
```
cd /opt/nvidia/deepstream/deepstream-4.0/sources/gst-plugins/gst-nvinfer
CUDA_VER=10.0 make and sudo CUDA_VER=10.0 make install
```

##

### Compiling edited model
1. Check your CUDA version (nvcc --version)
2. Open terminal
3. Go to your custom yolo directory
4. Type this command (example for CUDA 10.0 version):
```
CUDA_VER=10.0 make -C nvdsinfer_custom_impl_Yolo
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
To understand and edit deepstream_app_config file, read the [DeepStream SDK Development Guide - Configuration Groups](https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html#page/DeepStream%2520Development%2520Guide%2Fdeepstream_app_config.3.2.html)

In this repository have example of deepstream_app_config_yoloV3_tiny.txt file.

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
batched-push-timeout=-1
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

##

### Understanding and editing config_infer_primary
To understand and edit config_infer_primary file, read the [DeepStream SDK Development Guide - Application Customization](https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html#page/DeepStream%2520Development%2520Guide%2Fdeepstream_custom_model.html)

In this repository have example of config_infer_primary_yoloV3_tiny.txt file.

##

* Edit batch-size

```
# Number of sources
batch-size=1
```

##

* Edit num-detected-classes according number of classes in yolo.cfg

```
num-detected-classes=80
```

##

* Edit interval

```
# Interval of detection (keep >= 1 for real time detection on NVIDIA Jetson Nano)
interval=1
```

##

### Testing model
To run your custom yolo model, use this command (in your custom model directory; example for yolov3-tiny):
```
deepstream-app -c deepstream_app_config_yoloV3_tiny.txt
```

##

### Custom functions in your model

You can get metadata from deepstream in Python and C. For C, you need edit deepstream-app or deepstream-test code. For Python your need install and edit [this](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps).

You need manipulate [NvDsObjectMeta](https://docs.nvidia.com/metropolis/deepstream/4.0/dev-guide/DeepStream_Development_Guide/baggage/struct__NvDsObjectMeta.html), [NvDsFrameMeta](https://docs.nvidia.com/metropolis/deepstream/4.0/dev-guide/DeepStream_Development_Guide/baggage/struct__NvDsFrameMeta.html) and [NvOSD_RectParams](https://docs.nvidia.com/metropolis/deepstream/4.0/dev-guide/DeepStream_Development_Guide/baggage/struct__NvOSD__RectParams.html) to get label, position, etc. of bboxs.

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

##

I'm not an expert in DeepStream or Yolo, but I can help in any issue or question.

Sorry for any English error, it is not my native language.
