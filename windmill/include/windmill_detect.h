#ifndef MINDVERSIONCALL_WINDMILL_H
#define MINDVERSIONCALL_WINDMILL_H
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <queue>
#include <opencv2/opencv.hpp>
#include <memory>
#include <inference_engine.hpp>
#include <Eigen/Eigen>
#include "../../others/GlobalParams.h"

using namespace cv;
using namespace std;
using namespace InferenceEngine;

struct BuffObject
{
    Point2f apex[5];
    cv::Rect_<float> rect;
    int cls;
    int color;
    float prob;
    std::vector<cv::Point2f> pts;
};

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};
class BuffDetector
{
    public:
        BuffDetector();
        ~BuffDetector();
        bool detect(Mat &src,vector<BuffObject>& objects,Robotstatus &robotstatus);
        bool initModel(string path);
    private:
        Core ie;
        CNNNetwork network;                // 网络
        ExecutableNetwork executable_network;       // 可执行网络
        InferRequest infer_request;      // 推理请求
        MemoryBlob::CPtr moutput;
        string input_name;
        string output_name;
    
        Eigen::Matrix<float,3,3> transfrom_matrix;

};
#endif //MINDVERSIONCALL_WINDMILL_H
