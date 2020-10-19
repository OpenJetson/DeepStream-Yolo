# DeepStream YoloV4-Tiny
NVIDIA DeepStream SDK 5 configuration for YoloV4-Tiny model

Tested on NVIDIA Jetson Nano

##

* [Requirements](#requirements)
* [Convert Darknet model to ONNX model](#convert-darknet-model-to-onnx-model)
* [Convert ONNX model to TensorRT model](#convert-onnx-model-to-tensorrt-model)
* [Edit cpp file and compile lib](#edit-cpp-file-and-compile-lib)
* [Testing model](#testing-model)
* [FAQ](#faq)

##

### Requirements
* Python3
```
sudo apt-get install python3 python3-dev python3-pip
```

* Protobuf compiler
```
sudo apt-get install libprotobuf-dev protobuf-compiler
pip3 install protobuf
```

* Cython
```
pip3 install cython
```

* Numpy
```
pip3 install numpy
```

* Onnx
```
pip3 install onnx
```

* OnnxRuntime
```
pip3 install onnxruntime
```

* PyTorch
```
wget https://nvidia.box.com/shared/static/9eptse6jyly1ggt9axbja2yrmj6pbarc.whl -O torch-1.6.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev
pip3 install torch-1.6.0-cp36-cp36m-linux_aarch64.whl
```

##

### Convert Darknet model to ONNX model 
Thanks [Tianxiaomo](https://github.com/Tianxiaomo/pytorch-YOLOv4), [Ersheng](https://forums.developer.nvidia.com/t/get-wrong-infer-results-while-testing-yolov4-on-deepstream-5-0/125526/12) and [AlexeyAB](https://github.com/AlexeyAB/darknet).
1. Download repository
```
git clone https://github.com/Tianxiaomo/pytorch-YOLOv4.git
```

2. Download YoloV4-Tiny weights and cfg files to folder (or use your custom model)
```
cd pytorch-YOLOv4
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
```

3. Convert model
```
python3 demo_darknet2onnx.py yolov4-tiny.cfg yolov4-tiny.weights ./data/giraffe.jpg 1
```

3. Rename generated file to yolov4-tiny.onnx

##

### Convert ONNX model to TensorRT model
1. Convert model (exemple for workspace=1024 and FP16)
```
/usr/src/tensorrt/bin/trtexec --onnx=yolov4-tiny.onnx --explicitBatch --saveEngine=yolov4-tiny_fp16.engine --workspace=1024 --fp16
```

2. Move generated yolov4-tiny_fp16.engine file to your custom yolo folder

##

### Edit cpp file and compile lib
1. Add to following functions to nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp (or use my edited file available [here](https://github.com/marcoslucianops/DeepStream-Yolo/blob/master/examples/yolov4-tiny/nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp))
```
static NvDsInferParseObjectInfo convertBBoxYoloV4(const float& bx1, const float& by1, const float& bx2,
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

static void addBBoxProposalYoloV4(const float bx, const float by, const float bw, const float bh,
                     const uint& netW, const uint& netH, const int maxIndex,
                     const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
    NvDsInferParseObjectInfo bbi = convertBBoxYoloV4(bx, by, bw, bh, netW, netH);
    if (bbi.width < 1 || bbi.height < 1) return;

    bbi.detectionConfidence = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
}

static std::vector<NvDsInferParseObjectInfo>
decodeYoloV4Tensor(
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
            addBBoxProposalYoloV4(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, binfo);
        }

        bbox_location += 4;
        score_location += detectionParams.numClassesConfigured;
    }

    return binfo;
}

extern "C" bool NvDsInferParseCustomYoloV4(
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
        decodeYoloV4Tensor(
            (const float*)(boxes.buffer), (const float*)(scores.buffer), num_bboxes, detectionParams,
            networkInfo.width, networkInfo.height);

    objects.insert(objects.end(), outObjs.begin(), outObjs.end());

    objectList = objects;

    return true;
}
```

2. Locate your custom yolo folder and compile lib (example for dafault folder and CUDA 10.2)
```
cd /opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo/
CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo
```

##

### Testing model

Use my edited deepstream_app_config_yoloV4_tiny.txt and config_infer_primary_yoloV4_tiny.txt files available [here](https://github.com/marcoslucianops/DeepStream-Yolo/tree/master/examples/yolov4-tiny) (or edit these files for your custom model)

```
deepstream-app -c deepstream_app_config_yoloV4_tiny.txt
```

##

### FAQ
**Q:** Can I use YoloV4 on DeepStream?

**A:** Yes, you can. Do the same steps using YoloV4 files.

##

I'm not an expert in DeepStream or Yolo, but I can help in any issue or question.

Sorry for any English error, it is not my native language.
