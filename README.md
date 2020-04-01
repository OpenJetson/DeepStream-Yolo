# DeepStream-Yolo
NVIDIA DeepStream SDK configuration for YoloV3 model

Tested on NVIDIA Jetson Nano

##

* [Requirements](#requirements)
* [Editing default model](#editing-default-model)
* [Compiling edited model](#compiling-edited-model)
* [Editing yolo.cfg file](#editing-yolocfg-file)
* [Understanding and editing deepstream_app_config_yoloV3_tiny.txt](#understanding-and-editing-deepstream_app_config_yolov3_tinytxt)
* [Understanding and editing config_infer_primary_yoloV3_tiny.txt](#understanding-and-editing-config_infer_primary_yolov3_tinytxt)
* [Testing model](#testing-model)
* [FAQ](#faq)

##

### Requirements
* [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) >= 4
* [OpenCV](https://opencv.org/releases.html) (if you want to [populate confidence](#populate-confidence); it's built-in in latest [NVIDIA JetPack](https://developer.nvidia.com/embedded/jetpack))
* [Pre-treined YoloV3 model](https://github.com/AlexeyAB/darknet) (for NVIDIA Jetson Nano, I recommend YoloV3-Tiny-PRN (faster) or YoloV3-Tiny)

##

### Editing default model
1. Make a copy and remane objectDetector_Yolo folder (located in /opt/nvidia/deepstream/deepstream-4.0/sources/objectDetector_Yolo/) to your custom directory name.
2. Edit Yolo DeepStream for your custom model (in your custom directory), following this [Application Note](https://docs.nvidia.com/metropolis/deepstream/4.0/Custom_YOLO_Model_in_the_DeepStream_YOLO_App.pdf).
3. Copy your obj.names file to labels.txt in your custom yolo directory.
4. If you using **yolov3-tiny-prn** model, replace yolo.cpp to my edited yolo.cpp (available in nvdsinfer_custom_impl_Yolo folder on this repository).

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

##

### Compiling edited model
1. Check your CUDA version (nvcc --version)
2. Open terminal
3. Go to your custom directory
4. Type this command: (example for CUDA 10.0 version)
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

### Understanding and editing deepstream_app_config_yoloV3_tiny.txt
To understand and edit deepstream_app_config_yoloV3_tiny.txt file, read the [DeepStream SDK Development Guide - Configuration Groups](https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html#page/DeepStream%2520Development%2520Guide%2Fdeepstream_app_config.3.2.html)
* In this repository have example of deepstream_app_config_yoloV3_tiny.txt file.

##

### Understanding and editing config_infer_primary_yoloV3_tiny.txt
To understand and edit config_infer_primary_yoloV3_tiny.txt file, read the [DeepStream SDK Development Guide - Application Customization](https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html#page/DeepStream%2520Development%2520Guide%2Fdeepstream_custom_model.html)
* In this repository have example of config_infer_primary_yoloV3_tiny.txt file.

##

### Testing model
To run your custom yolo model, use this command (in your custom model directory):
```
deepstream-app -c deepstream_app_config_yoloV3_tiny.txt
```

##

### FAQ
**Q:** Can I run custom yolo model on deepstream with non-square shape?

**A:** You can, but the accuracy will greatly decrease. If you want to test, see [this patch](https://forums.developer.nvidia.com/t/trouble-in-converting-non-square-grid-in-yolo-network-to-tensorrt-via-deepstream/107541/12)

<br>

**Q:** How to make more than 1 yolo inference?

**A:** See MULTIPLE-INFERENCES.md in this repository (comming soon)

##

I'm not an expert in DeepStream or Yolo, but I can help in any issue or question.
