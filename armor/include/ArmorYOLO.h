#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <openvino/openvino.hpp>
#include <inference_engine.hpp>

using namespace std;
using namespace InferenceEngine;
using namespace cv;
#define IMG_SIZE  416  //推理图像大小,GMaster团队使用416x416作为图像推理大小
#define DEVICE "GPU" // 设备选择，Inter NUC13 应使用GNA，其他使用GPU
//#define VIDEO //是否展示推理视频
#define CLS_NUM 14
#define NMS_THRESHOLD 0.25
#define CONF_THRESHOLD 0.5
#define SCORE_THRESHOLD 0.3

class armor_detect_yolo {
public:
    armor_detect_yolo();

    struct Object {
        cv::Rect_<float> rect;
        int label;
        float prob;
        cv::Point2f point[4];
    };
    struct Resize {
        cv::Mat resized_image;
        int dw;
        int dh;
    };

    bool work(cv::Mat src_img,vector<Object>& object);

private:
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    ov::Tensor input_tensor;
    const std::vector<std::string> class_names = {
            "B1", "B2", "B3", "B4", "B5", "BO", "BS", "R1", "R2", "R3", "R4", "R5", "RO", "RS"
    };

    static float sigmoid(float x) {
        return static_cast<float>(1.f / (1.f + exp(-x)));
    }

};
