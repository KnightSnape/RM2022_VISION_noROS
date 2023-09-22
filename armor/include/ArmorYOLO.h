#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <openvino/openvino.hpp>
#include <inference_engine.hpp>

class armor_detect_yolo
{
    public:
        armor_detect_yolo();

        struct Object
        {
            cv::Rect_<float> rect;
            int label;
            float prob;
            cv::Point2f point[4];
        };

    void load_param();

    cv::Mat letterbox(cv::Mat &src, int h, int w, std::vector<float> &padd);

    cv::Rect scale_box(cv::Rect box, std::vector<float> &padd);

    void drawPred(int classId, float conf, cv::Rect box, float ratio, float raw_h, float raw_w, cv::Mat &frame,
                  const std::vector<std::string> &classes);

    void generate_proposals(int stride, const float *feat, float prob_threshold, std::vector<Object> &objects);

    bool work(cv::Mat src_img,std::vector<Object>& objects);

    private:

        cv::TickMeter meter;

        ov::Core core;

        std::shared_ptr<ov::Model> model;

        ov::CompiledModel compiled_model;

        ov::InferRequest infer_request;

        ov::Tensor input_tensor1;

        const std::vector<std::string> class_names = {
            "B1","B2","B3","B4","B5","BO","BS","R1","R2","R3","R4","R5","RO","RS"
        };

        inline float sigmoid(float x)
        {
            return static_cast<float>(1.f / (1.f + exp(-x)));
        }

};
