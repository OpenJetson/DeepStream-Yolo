# DeepStream YoloV5
NVIDIA DeepStream SDK 5 configuration for YoloV5s

Example for YoloV5s (Tested on NVIDIA Jetson Nano)

Thanks [DanaHan](https://github.com/DanaHan/Yolov5-in-Deepstream-5.0), [wang-xinyu](https://github.com/wang-xinyu/tensorrtx) and [Ultralytics](https://github.com/ultralytics/yolov5)

##

Sample video running in FP16 mode (AVG FPS: 9.28): https://youtu.be/IAK-u6Euy_Q

Sample video running in FP32 mode (AVG FPS: 7.92): https://youtu.be/fbOvel0eQMU

##

* [Requirements](#requirements)
* [Convert PyTorch model to wts file](#convert-pytorch-model-to-wts-file)
* [Convert wts file to TensorRT model](#convert-wts-file-to-tensorrt-model)
* [Edit cpp file and compile lib](#edit-cpp-file-and-compile-lib)
* [Testing model](#testing-model)
* [FAQ](#faq)

##

### Requirements
* Python3
```
sudo apt-get install python3 python3-dev python3-pip
```

* OpenCV
```
sudo apt-get install libopencv-python
```

* Matplotlib
```
pip3 install matplotlib
```

* Scipy
```
pip3 install scipy
```

* tqdm
```
pip3 install tqdm
```

* PyTorch (for Jetson platform)
```
wget https://nvidia.box.com/shared/static/9eptse6jyly1ggt9axbja2yrmj6pbarc.whl -O torch-1.6.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev
pip3 install torch-1.6.0-cp36-cp36m-linux_aarch64.whl
```

* TorchVision (for Jetson platform)
```
git clone -b v0.7.0 https://github.com/pytorch/vision torchvision
sudo apt-get install libjpeg-dev zlib1g-dev python3-pip
cd torchvision
export BUILD_VERSION=0.7.0
sudo python3 setup.py install
```

##

### Convert PyTorch model to wts file
1. Download repositories
```
git clone https://github.com/DanaHan/Yolov5-in-Deepstream-5.0.git yolov5converter
git clone https://github.com/wang-xinyu/tensorrtx.git
git clone https://github.com/ultralytics/yolov5.git
```

2. Download YoloV5s weights to yolov5/weights directory
```
wget https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5s.pt -P yolov5/weights/
```

3. Copy gen_wts.py file (from tensorrtx/yolov5 folder) to yolov5 (ultralytics) folder
```
cp tensorrtx/yolov5/gen_wts.py yolov5/gen_wts.py
```

4. Generate wts file
```
cd yolov5
python3 gen_wts.py
```

yolov5s.wts file will be generated in yolov5 folder

##

### Convert wts file to TensorRT model
1. Replace yololayer files from tensorrtx/yolov5 folder to yololayer and hardswish files from yolov5converter
```
mv yolov5converter/yololayer.cu tensorrtx/yolov5/yololayer.cu
mv yolov5converter/yololayer.h tensorrtx/yolov5/yololayer.h
mv yolov5converter/hardswish.cu tensorrtx/yolov5/hardswish.cu
mv yolov5converter/hardswish.h tensorrtx/yolov5/hardswish.h
```

2. Move generated yolov5s.wts file to tensorrtx/yolov5 folder
```
cp yolov5/yolov5s.wts tensorrtx/yolov5/yolov5s.wts
```

3. Build tensorrtx/yolov5
```
cd tensorrtx/yolov5
mkdir build
cd build
cmake ..
make
```

4. Convert to TensorRT model (yolov5s.engine and libmyplugins.so files will be generated in tensorrtx/yolov5/build directory)
```
sudo ./yolov5 -s
```

5. Copy generated files to custom yolo folder (example for dafault yolo folder)
```
cp yolov5s.engine /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo/yolov5s.engine
cp libmyplugins.so /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo/libmyplugins.so
```

##

### Edit cpp file and compile lib
1. Add to following functions to nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp located in your deepstream-yolo folder (or use my edited file available [here](https://github.com/marcoslucianops/DeepStream-Yolo/blob/master/examples/yolov5s/nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp))
```
#define NMS_THRESH 0.45
#define CONF_THRESH 0.25

extern "C" bool NvDsInferParseCustomYoloV5(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);


static constexpr int LOCATIONS = 4;
struct alignas(float) Detection{
        //center_x center_y w h
        float bbox[LOCATIONS];
        float conf;  // bbox_conf * cls_conf
        float class_id;
    };

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
        std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
        std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
        std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
}

bool cmp(Detection& a, Detection& b) {
    return a.conf > b.conf;
}

void nms(std::vector<Detection>& res, float *output, float conf_thresh, float nms_thresh = 0.45) {
    int det_size = sizeof(Detection) / sizeof(float);
    std::map<float, std::vector<Detection>> m;
    for (int i = 0; i < output[0] && i < 1000; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }
}

static bool NvDsInferParseYoloV5(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
    }

    std::vector<Detection> res;

    nms(res, (float*)(outputLayersInfo[0].buffer), CONF_THRESH, NMS_THRESH);
    //std::cout<<"Nms done sucessfully----"<<std::endl;
    
    for(auto& r : res) {
	    NvDsInferParseObjectInfo oinfo;        
        
	    oinfo.classId = r.class_id;
	    oinfo.left    = static_cast<unsigned int>(r.bbox[0]-r.bbox[2]*0.5f);
	    oinfo.top     = static_cast<unsigned int>(r.bbox[1]-r.bbox[3]*0.5f);
	    oinfo.width   = static_cast<unsigned int>(r.bbox[2]);
	    oinfo.height  = static_cast<unsigned int>(r.bbox[3]);
	    oinfo.detectionConfidence = r.conf;
        //std::cout << static_cast<unsigned int>(r.bbox[0]) << "," << static_cast<unsigned int>(r.bbox[1]) << "," << static_cast<unsigned int>(r.bbox[2]) << "," 
        //          << static_cast<unsigned int>(r.bbox[3]) << "," << static_cast<unsigned int>(r.class_id) << "," << static_cast<unsigned int>(r.conf) << std::endl;
	    objectList.push_back(oinfo);        
    }
    
    return true;
}

extern "C" bool NvDsInferParseCustomYoloV5(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    return NvDsInferParseYoloV5 (
        outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV5);
```

2. Locate your custom yolo folder and compile lib (example for dafault yolo folder and CUDA 10.2)
```
cd /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo/
CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo
```

##

### Testing model

Use my edited deepstream_app_config_yoloV5s.txt and config_infer_primary_yoloV5s.txt files available [here](https://github.com/marcoslucianops/DeepStream-Yolo/tree/master/examples/yolov5s) (or edit these files for your custom model)

```
LD_PRELOAD=./libmyplugins.so deepstream-app -c deepstream_app_config_yoloV5s.txt
```

##

### FAQ
**Q:** Can I use others YoloV5 models (YoloV5m, YoloV5l, YoloV5x) on DeepStream?

**A:** You can. See this [link](https://github.com/wang-xinyu/tensorrtx/blob/master/yolov5/README.md#config) and do the same steps using your selected YoloV5 file. Remember to edit deepstream_app_config_yoloV5_tiny.txt and config_infer_primary_yoloV5_tiny.txt files to your selected YoloV5.
