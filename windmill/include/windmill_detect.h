#pragma once
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

#define NMS_THRESHOLD 0.25f
#define CONF_THRESHOLD 0.45f
#define IMG_SIZE 416
#define ANCHOR 3 //anchor 数量
#define DEVICE "GPU"
//#define VIDEO
#define KPT_NUM 5
#define CLS_NUM 4

class buff_kpt
{
    public:
        buff_kpt();

        struct Object {
            cv::Rect_<float> rect;
            int label;
            float prob;
            std::vector<cv::Point2f> kpt;
            cv::Point2f center;
            cv::Point2f center_R;
        };
        void load_params(std::string model_path);

        cv::Mat letter_box(cv::Mat &src, int h, int w, std::vector<float> &padd);

        std::vector<cv::Point2f>
        scale_box_kpt(std::vector<cv::Point2f> points, std::vector<float> &padd, float raw_w, float raw_h, int idx);

        cv::Rect scale_box(cv::Rect box, std::vector<float> &padd, float raw_w, float raw_h);

        void drawPred(int classId, float conf, cv::Rect box, std::vector<cv::Point2f> point, cv::Mat &frame,
                    const std::vector<std::string> &classes);

        static void generate_proposals(int stride, const float *feat, std::vector<Object> &objects);

        std::vector<Object> work(cv::Mat src_img);

    private:
        ov::Core core;
        std::shared_ptr<ov::Model> model;
        ov::CompiledModel compiled_model;
        ov::InferRequest infer_request;
        ov::Tensor input_tensor1;

        const std::vector<std::string> class_names = {
             "RR", "RW", "BR", "BW"
        };

        static float sigmoid(float x)
        {
            return static_cast<float>(1.f / (1.f + exp(-x)));
        }
};
