# Multiple inferences

For multiple inferences (primary gie, secondary gie, etc.) is necessary do some implementations on sample yolo configs files.

1. Make a folder named yolo in /opt/nvidia/deepstream/deepstream-4.0/sources/ directory.
2. Make a folder, in created yolo directory, named pgie (where you will put files of primary inference).
3. Make a folder, for each secondary inference, in created yolo directory, named sgie* (* = 1, 2, 3, etc.; depending on the number of secondary inferences; where you will put files of another inferences).
4. Copy of nvdsinfer_custom_impl_Yolo folder (located in /opt/nvidia/deepstream/deepstream-4.0/sources/objectDetector_Yolo) to each inference directory (pgie, sgie*).
5. Edit Yolo DeepStream for your custom model (in each inference directory: pgie, sgie*), according each yolo.cfg (v3, v3-tiny, etc.) file, following this [Application Note](https://docs.nvidia.com/metropolis/deepstream/4.0/Custom_YOLO_Model_in_the_DeepStream_YOLO_App.pdf).
6. Copy and remane each obj.names file to labels.txt in each inference directory (pgie, sgie*), according each inference type.
7. Copy your yolo.cfg (v3, v3-tiny, etc.) file to each inference directory (pgie, sgie*), according each inference type.
8. Copy only config_infer_primary.txt (same of your yolo model; v3, v3-tiny, etc.) or download my edited file (in examples folder, on this repository, if you using yolov3-tiny) to each custom inference directory (pgie, sgie*).
9. Copy only deepstream_app_config.txt (same of your yolo model; v3, v3-tiny, etc.) or download my edited file (in examples folder, on this repository) to created yolo directory.

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

