# Multiple inferences

To multiple inferences (primary gie, secondary gie, etc.) is necessary do some implementations on sample yolo configs files.

1. Make a folder named yolo in /opt/nvidia/deepstream/deepstream-4.0/sources/ directory.
2. Make a folder, in created yolo directory, named pgie (where you will put files of primary inference).
3. Make a folder, for each secondary inference, in created yolo directory, named sgie* (* = 1, 2, 3, etc.; depending on the number of secondary inferences; where you will put files of another inferences).
4. Copy of nvdsinfer_custom_impl_Yolo folder (located in /opt/nvidia/deepstream/deepstream-4.0/sources/objectDetector_Yolo) to each inference directory (pgie, sgie*).
5. Edit Yolo DeepStream for your custom model (in each inference directory: pgie, sgie*), according each yolo.cfg (v3, v3-tiny, etc.) file, following this [Application Note](https://docs.nvidia.com/metropolis/deepstream/4.0/Custom_YOLO_Model_in_the_DeepStream_YOLO_App.pdf).
6. Copy and remane each obj.names file to labels.txt in each inference directory (pgie, sgie*), according each inference type.
7. Copy your yolo.cfg (v3, v3-tiny, etc.) file to each inference directory (pgie, sgie*), according each inference type.
8. Copy config_infer_primary.txt (same of your yolo model; v3, v3-tiny, etc.) from /opt/nvidia/deepstream/deepstream-4.0/sources/objectDetector_Yolo to created yolo directory.
8. Copy and rename config_infer_primary.txt (same of your yolo model; v3, v3-tiny, etc.) from /opt/nvidia/deepstream/deepstream-4.0/sources/objectDetector_Yolo to each config_infer_secondary*.txt (* = 1, 2, 3, etc.; depending on the number of secondary inferences) to created yolo directory.
9. Copy deepstream_app_config.txt (same of your yolo model; v3, v3-tiny, etc.) from /opt/nvidia/deepstream/deepstream-4.0/sources/objectDetector_Yolo to created yolo directory.

In example folder, in this repository, have Makefile, config_infer and deepstream_app_config example files for yolov3-tiny.

##

### Editing nvdsinfer_context_impl.cpp
To change folder where yolo.engine will be generated, is necessary to edit nvdsinfer_context_impl.cpp (lines 1672-1674) in /opt/nvidia/deepstream/deepstream-4.0/sources/libs/nvdsinfer directory.

Before:
```
            char *cwd = getcwd(NULL, 0);
            engineFileName << cwd << "/model";
            free(cwd);
```
After:
```
            std::string s = initParams.modelEngineFilePath;
            size_t pos = s.rfind("/");
            s.erase(pos);
            //char *cwd = getcwd(NULL, 0);
            engineFileName << s << "/model";
            //free(cwd);
```
And compile it again (requires libopencv-dev; example for CUDA 10.0 version):
```
cd /opt/nvidia/deepstream/deepstream-4.0/sources/libs/nvdsinfer
CUDA_VER=10.0 make and sudo CUDA_VER=10.0 make install
```

##

### Editing Makefile
To compile nvdsinfer_custom_impl_Yolo without errors is necessary to edit Makefile (line 28), in nvdsinfer_custom_impl_Yolo folder in each inference directory.
```
CFLAGS+= -I../../includes -I/usr/local/cuda-$(CUDA_VER)/include
```
To:
```
CFLAGS+= -I../../../includes -I/usr/local/cuda-$(CUDA_VER)/include
```

##

### Compiling edited models
1. Open terminal.
2. Go to inference directory.
3. Type this command (example for CUDA 10.0 version):
```
CUDA_VER=10.0 make -C nvdsinfer_custom_impl_Yolo
```
**Do this for each inference!**

##

### Editing yolo.cfg file
Set batch=1 and subdivisions=1 in each yolo.cfg file (in each inference directory: pgie, sgie*) 
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

### Editing deepstream_app_config

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
config-file=pgie/config_infer_primary_yoloV3_tiny.txt
```

##

* Add secondary-gie

Example for 1 secondary-gie (2 inferences):
```
[secondary-gie0]
enable=1
gpu-id=0
gie-unique-id=2
# If you want secodary inference operate on specified gie id (gie-unique-id you want to operate: 1, 2, etc.)
operate-on-gie-id=1
# If you want secodary inference operate on specified class ids of gie (class ids you want to operate: 1, 1;2, 2;3;4, 3 etc.)
operate-on-class-ids=0
nvbuf-memory-type=0
config-file=sgie1/config_infer_secondary1_yoloV3_tiny.txt
```
Example for 2 secondary-gie (3 inferences):
```
[secondary-gie0]
enable=1
gpu-id=0
gie-unique-id=2
operate-on-gie-id=1
operate-on-class-ids=0
nvbuf-memory-type=0
config-file=sgie1/config_infer_secondary1_yoloV3_tiny.txt

[secondary-gie1]
enable=1
gpu-id=0
gie-unique-id=3
operate-on-gie-id=1
operate-on-class-ids=0
nvbuf-memory-type=0
config-file=sgie2/config_infer_secondary2_yoloV3_tiny.txt
```

##

### Editing config_infer

* Edit path of files

Example for primary (using fp16):
```
custom-network-config=pgie/yolov3-tiny.cfg
model-file=yolov3-tiny.weights
model-engine-file=model_b1_fp16.engine
labelfile-path=labels.txt
custom-lib-path=nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
```
Example for secondary1 (using fp16):
```
custom-network-config=sgie1/yolov3-tiny.cfg
model-file=yolov3-tiny.weights
model-engine-file=model_b16_fp16.engine
labelfile-path=labels.txt
custom-lib-path=nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
```
Example for secondary2 (using fp16):
```
custom-network-config=sgie2/yolov3-tiny.cfg
model-file=yolov3-tiny.weights
model-engine-file=model_b16_fp16.engine
labelfile-path=labels.txt
custom-lib-path=nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
```

##

* Edit gie-unique-id

Example for primary:
```
gie-unique-id=1
process-mode=1
```
Example for secondary1:
```
gie-unique-id=2
process-mode=2
```
Example for secondary2:
```
gie-unique-id=3
process-mode=3
```

##

* Edit batch-size

Example for primary:
```
# Number of sources
batch-size=1
```
Example for all secondary:
```
batch-size=16
```

##

* If you want secodary inference operate on specified gie id

```
#gie-unique-id you want to operate (1, 2, etc.)
operate-on-gie-id=1
```

##

* If you want secodary inference operate on specified class ids of gie

```
#class ids you want to operate (1, 1;2, 2;3;4, 3 etc.)
operate-on-class-ids=0
```

##

* Edit num-detected-classes of each config_infer file according number of classes in each yolo.cfg

```
num-detected-classes=80
```

##

* Edit interval in config_infer_primary file

```
# Interval of detection (keep >= 1 for real time detection on NVIDIA Jetson Nano)
interval=1
```

##

### Testing model
To run your custom yolo model, use this command (in your custom yolo model directory; example for yolov3-tiny):
```
deepstream-app -c deepstream_app_config_yoloV3_tiny.txt
```

##

I'm not an expert in DeepStream or Yolo, but I can help in any issue or question.
