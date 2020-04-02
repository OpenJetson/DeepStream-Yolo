# DeepStream-Yolo
NVIDIA DeepStream SDK configuration for Yolo model

Tested on NVIDIA Jetson Nano

##

* [Requirements](#requirements)
* [Editing default model](#editing-default-model)
* [Compiling edited model](#compiling-edited-model)
* [Editing yolo.cfg file](#editing-yolocfg-file)
* [Understanding and editing deepstream_app_config](#understanding-and-editing-deepstream_app_config)
* [Understanding and editing config_infer_primary](#understanding-and-editing-config_infer_primary)
* [Testing model](#testing-model)
* [FAQ](#faq)

##

### Requirements
* [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) >= 4
* [OpenCV](https://opencv.org/releases.html) (if you want to [populate confidence](#populate-confidence); it's built-in in latest [NVIDIA JetPack](https://developer.nvidia.com/embedded/jetpack); sudo apt-get install libopencv-dev)
* [GStreamer-1.0](https://gstreamer.freedesktop.org/) Development package and Base Plugins Development package (sudo apt-get install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev)
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
* In this repository have example of deepstream_app_config_yoloV3_tiny.txt file.

##

### Understanding and editing config_infer_primary
To understand and edit config_infer_primary file, read the [DeepStream SDK Development Guide - Application Customization](https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html#page/DeepStream%2520Development%2520Guide%2Fdeepstream_custom_model.html)
* In this repository have example of config_infer_primary_yoloV3_tiny.txt file.

##

### Testing model
To run your custom yolo model, use this command (in your custom model directory; example for yolov3-tiny):
```
deepstream-app -c deepstream_app_config_yoloV3_tiny.txt
```

##

### FAQ
**Q:** Can I run custom yolo model on deepstream with non-square shape?

**A:** You can, but the accuracy will greatly decrease. If you want to test, see [this patch](https://forums.developer.nvidia.com/t/trouble-in-converting-non-square-grid-in-yolo-network-to-tensorrt-via-deepstream/107541/12).

<br>

**Q:** How to make more than 1 yolo inference?

**A:** See [MULTIPLE-INFERENCES.md](https://github.com/marcoslucianops/DeepStream-Yolo/blob/master/MULTIPLE-INFERENCES.md) in this repository.
##

I'm not an expert in DeepStream or Yolo, but I can help in any issue or question.
