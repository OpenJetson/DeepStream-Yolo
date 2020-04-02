# Multiple inferences

For multiple inferences (primary gie, secondary gie, etc.) is necessary do some implementations on sample yolo configs files.

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

In example folder, in this repository, have config_infer and deepstream_app_config example files for yolov3-tiny.

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
And compile it again (requires libopencv):
```
cd /opt/nvidia/deepstream/deepstream-4.0/sources/libs/nvdsinfer
make and sudo make install
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

## Editing config_infer
* In each config_infer (primary, secondary), edit path of files.
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
# Number of streams
batch-size=1
```
Example for all secondary:
```
batch-size=16
```

##

* If you want secodary inference operate on specified inference and specified class ids
```
#gie-unique-id you want to operate (1, 2, etc.)
operate-on-gie-id=1
```

##

* If you want secodary inference operate on specified class ids
```
#class ids you want to operate (1, 1;2, 2;3;4, 3 etc.)
operate-on-class-ids=0
```

##

* Edit num-detected-classes of each config_infer file according number of classes in each yolo.cfg
```
num-detected-classes=80
```

Basically, you need to make these modifications.


To be continued..
