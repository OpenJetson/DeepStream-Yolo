# DeepStream YoloV5s
NVIDIA DeepStream SDK 5 configuration for YoloV5s

Tested on NVIDIA Jetson Nano

Thanks [DanaHan](https://github.com/DanaHan/Yolov5-in-Deepstream-5.0), [wang-xinyu](https://github.com/wang-xinyu/tensorrtx) and [Ultralytics](https://github.com/ultralytics/yolov5)

##

* [Requirements](#requirements)
* [Convert PyTorch model to wts file](#pytorch-model-to-wts-file)
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

* Cython
```
pip3 install cython
```

* Numpy
```
pip3 install numpy
```

* Matplotlib
```
pip3 install matplotlib
```

* Scipy
```
pip3 install scipy
```

* Pillow
```
pip3 install pillow
```

* tqdm
```
pip3 install tqdm
```

* PyTorch
```
wget https://nvidia.box.com/shared/static/9eptse6jyly1ggt9axbja2yrmj6pbarc.whl -O torch-1.6.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev
pip3 install torch-1.6.0-cp36-cp36m-linux_aarch64.whl
```

* TorchVision
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
git clone https://github.com/ultralytics/yolov5.git
```

2. Download YoloV5s weights (or use your custom model) to yolov5/weights directory
```
wget https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5s.pt -P yolov5/weights/
```

3. Copy gen_wts.py file (from yolov5converter folder) to yolov5 (ultralytics) folder
```
cp yolov5converter/gen_wts.py yolov5/gen_wts.py
```

4. Generate wts file
```
cd yolov5
python3 gen_wts.py
```

yolov5s.wts file will be generated in yolov5 folder

##

### Convert wts file to TensorRT model
1. Move generated yolov5s.wts file to yolov5converter folder
```
cp yolov5/yolov5s.wts yolov5converter/yolov5s.wts
```

1. Build yolov5converter
```
cd yolov5converter
mkdir build
cd build
cmake ..
make
```

3. Convert to TensorRT model (yolov5s.engine and libmyplugins.so files will be generated in yolov5converter/build directory)
```
sudo ./yolov5 -s
```

4. Copy generated files to custom yolo folder (example for dafault yolo folder)
```
cp yolov5s.engine /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo/yolov5s.engine
cp libmyplugins.so /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo/libmyplugins.so
```

##

### Edit cpp file and compile lib
1. Add to following functions to nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp located in your deepstream-yolo folder (or use my edited file available [here](https://github.com/marcoslucianops/DeepStream-Yolo/blob/master/examples/yolov5s/nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp))
```
static NvDsInferParseObjectInfo convertBBoxYoloV5(const float& bx1, const float& by1, const float& bx2,
                                     const float& by2, const uint& netW, const uint& netH)
{
    NvDsInferParseObjectInfo b;
    // Restore coordinates to network input resolution

    float x1 = bx1 * netW;
    float y1 = by1 * netH;
    float x2 = bx2 * netW;
    float y2 = by2 * netH;

    x1 = clamp(x1, 0, netW);
    y1 = clamp(y1, 0, netH);
    x2 = clamp(x2, 0, netW);
    y2 = clamp(y2, 0, netH);

    b.left = x1;
    b.width = clamp(x2 - x1, 0, netW);
    b.top = y1;
    b.height = clamp(y2 - y1, 0, netH);

    return b;
}

static void addBBoxProposalYoloV5(const float bx, const float by, const float bw, const float bh,
                     const uint& netW, const uint& netH, const int maxIndex,
                     const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
    NvDsInferParseObjectInfo bbi = convertBBoxYoloV5(bx, by, bw, bh, netW, netH);
    if (bbi.width < 1 || bbi.height < 1) return;

    bbi.detectionConfidence = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
}

static std::vector<NvDsInferParseObjectInfo>
decodeYoloV5Tensor(
    const float* boxes, const float* scores,
    const uint num_bboxes, NvDsInferParseDetectionParams const& detectionParams,
    const uint& netW, const uint& netH)
{
    std::vector<NvDsInferParseObjectInfo> binfo;

    uint bbox_location = 0;
    uint score_location = 0;
    for (uint b = 0; b < num_bboxes; ++b)
    {
        float bx1 = boxes[bbox_location];
        float by1 = boxes[bbox_location + 1];
        float bx2 = boxes[bbox_location + 2];
        float by2 = boxes[bbox_location + 3];

        float maxProb = 0.0f;
        int maxIndex = -1;

        for (uint c = 0; c < detectionParams.numClassesConfigured; ++c)
        {
            float prob = scores[score_location + c];
            if (prob > maxProb)
            {
                maxProb = prob;
                maxIndex = c;
            }
        }

        if (maxProb > detectionParams.perClassPreclusterThreshold[maxIndex])
        {
            addBBoxProposalYoloV5(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, binfo);
        }

        bbox_location += 4;
        score_location += detectionParams.numClassesConfigured;
    }

    return binfo;
}

extern "C" bool NvDsInferParseCustomYoloV5(
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

    std::vector<NvDsInferParseObjectInfo> objects;

    const NvDsInferLayerInfo &boxes = outputLayersInfo[0]; // num_boxes x 4
    const NvDsInferLayerInfo &scores = outputLayersInfo[1]; // num_boxes x num_classes

    // 3 dimensional: [num_boxes, 1, 4]
    assert(boxes.inferDims.numDims == 3);
    // 2 dimensional: [num_boxes, num_classes]
    assert(scores.inferDims.numDims == 2);

    // The second dimension should be num_classes
    assert(detectionParams.numClassesConfigured == scores.inferDims.d[1]);
    
    uint num_bboxes = boxes.inferDims.d[0];

    // std::cout << "Network Info: " << networkInfo.height << "  " << networkInfo.width << std::endl;

    std::vector<NvDsInferParseObjectInfo> outObjs =
        decodeYoloV5Tensor(
            (const float*)(boxes.buffer), (const float*)(scores.buffer), num_bboxes, detectionParams,
            networkInfo.width, networkInfo.height);

    objects.insert(objects.end(), outObjs.begin(), outObjs.end());

    objectList = objects;

    return true;
}
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

**A:** I think you can. Do the same steps using your selected YoloV5 file. Remember to edit deepstream_app_config_yoloV5_tiny.txt and config_infer_primary_yoloV5_tiny.txt files to your selected YoloV5.

##

I'm not an expert in DeepStream or Yolo, but I can help in any issue or question.

Sorry for any English error, it is not my native language.
