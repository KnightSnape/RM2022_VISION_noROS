#ifndef GMASTER_CV_2023_ARMORYOLO_H
#define GMASTER_CV_2023_ARMORYOLO_H

#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <sys/types.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <openvino/openvino.hpp>
#include <inference_engine.hpp>
using namespace InferenceEngine;

#define NMS_THRESHOLD 0.20f //NMS参数
#define CONF_THRESHOLD 0.70f //置信度参数
#define IMG_SIZE  416  // 推理图像大小，如果不是640 和 416 需要自己在下面添加anchor
#define DETECT_MODE 2 //Pose 0 Win_Kpt 1 Detect 2
#define DEVICE "CPU" // 设备选择
//#define VIDEO //是否展示推理视频

#if DETECT_MODE == 0 // 装甲板四点模型
#define KPT_NUM 17
#define CLS_NUM 1
#define MODEL_PATH "/home/knight/C++_inference_openvino_kpt/yolov8n-pose.onnx"
#elif DETECT_MODE == 1 // 能量机关五点模型
#define KPT_NUM 5
#define CLS_NUM 4
#define MODEL_PATH "/home/zr/Downloads/best_win_416.onnx"
#elif DETECT_MODE == 2 // 视觉识别版检测模型
#define KPT_NUM 0
#define CLS_NUM 14
#define MODEL_PATH "../config/armor_0607.xml"
#endif
extern float last_conf;
class yolo_kpt {
public:
    yolo_kpt();

    struct Object {
        cv::Rect_<float> rect;
        int label;
        float prob;
        std::vector<cv::Point2f> kpt;
    };
    cv::Mat letter_box(cv::Mat &src, int h, int w, std::vector<float> &padd);

    std::vector<cv::Point2f>
    scale_box_kpt(std::vector<cv::Point2f> points, std::vector<float> &padd, float raw_w, float raw_h, int idx);

    cv::Rect scale_box(cv::Rect box, std::vector<float> &padd, float raw_w, float raw_h);

    void drawPred(int classId, float conf, cv::Rect box, std::vector<cv::Point2f> point, cv::Mat &frame,
                  const std::vector<std::string> &classes);

    static void generate_proposals(const float *feat, std::vector<Object> &objects,ov::Shape &output_shape);

    std::vector<Object> work(cv::Mat src_img);

private:
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    ov::Tensor input_tensor1;
#if DETECT_MODE == 0
    const std::vector<std::string> class_names = {
            "person" //for yolov8-n-pose
//             "B1", "B2", "B3", "B4", "B5", "BO", "BS", "R1", "R2", "R3", "R4", "R5", "RO", "RS"
    };
#elif DETECT_MODE == 1
    const std::vector<std::string> class_names = {
             "RR", "RW", "BR", "BW"
    };
#elif DETECT_MODE == 2
     const std::vector<std::string> class_names = {
             "B1", "B2", "B3", "B4", "B5", "BO", "BS", "R1", "R2", "R3", "R4", "R5", "RO", "RS"
    };
#endif

    static float sigmoid(float x) {
        return static_cast<float>(1.f / (1.f + exp(-x)));
    }

};

#endif //GMASTER_CV_2023_ARMORYOLO_H
