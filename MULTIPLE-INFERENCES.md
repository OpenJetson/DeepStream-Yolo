# Multiple inferences

1. Make a folder named yolo in /opt/nvidia/deepstream/deepstream-5.0/sources/ directory.
2. Make a folder, in created yolo directory, named pgie (where you will put files of primary inference).
3. Make a folder, for each secondary inference, in created yolo directory, named sgie* (* = 1, 2, 3, etc.; depending on the number of secondary inferences; where you will put files of another inferences).
4. Copy of nvdsinfer_custom_impl_Yolo folder (located in /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo) to each inference directory (pgie, sgie*).
5. Edit Yolo DeepStream for your custom model (in each inference directory: pgie, sgie*), according each yolo.cfg (v3, v3-tiny, etc.) file: https://github.com/marcoslucianops/DeepStream-Yolo#editing-default-model
6. Copy and remane each obj.names file to labels.txt in each inference directory (pgie, sgie*), according each inference type.
7. Copy your yolo.cfg (v3, v3-tiny, etc.) file to each inference directory (pgie, sgie*), according each inference type.
8. Copy config_infer_primary.txt (same of your yolo model; v3, v3-tiny, etc.) from /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo to created yolo directory.
8. Copy and rename config_infer_primary.txt (same of your yolo model; v3, v3-tiny, etc.) from /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo to each config_infer_secondary*.txt (* = 1, 2, 3, etc.; depending on the number of secondary inferences) to created yolo directory.
9. Copy deepstream_app_config.txt (same of your yolo model; v3, v3-tiny, etc.) from /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo to created yolo directory.

In example folder, in this repository, have Makefile, config_infer and deepstream_app_config example files for YoloV3-Tiny.

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
3. Type this command (example for CUDA 10.2 version):
```
CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo
```

**Do this for each gie!**

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

### Add secondary-gie to deepstream_app_config after primary-gie

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

* Edit path of config files

Example for primary

```
custom-network-config=pgie/yolov3-tiny.cfg
```

Example for secondary1

```
custom-network-config=sgie1/yolov3-tiny.cfg
```

Example for secondary2

```
custom-network-config=sgie2/yolov3-tiny.cfg
```

##

* Edit gie-unique-id

Example for primary

```
gie-unique-id=1
process-mode=1
```

Example for secondary1

```
gie-unique-id=2
process-mode=2
```

Example for secondary2

```
gie-unique-id=3
process-mode=2
```

##

* Edit batch-size

Example for primary

```
# Number of sources
batch-size=1
```

Example for all secondary:

```
batch-size=16
```

##

* If you want secodary inference operate on specified gie id (gie-unique-id you want to operate: 1, 2, etc.)

```
operate-on-gie-id=1
```

##

* If you want secodary inference operate on specified class ids of gie (class ids you want to operate: 1, 1;2, 2;3;4, 3 etc.)

```
operate-on-class-ids=0
```

### Testing model
To run your custom yolo model, use this command (in your custom yolo model directory; example for yolov3-tiny):

```
deepstream-app -c deepstream_app_config_yoloV3_tiny.txt
```

**During test process, engine file will be generated. When engine build process is done, move engine file to respective gie folder (pgie, sgie1, etc.)**

##

I'm not an expert in DeepStream or Yolo, but I can help in any issue or question.

Sorry for any English error, it is not my native language.
