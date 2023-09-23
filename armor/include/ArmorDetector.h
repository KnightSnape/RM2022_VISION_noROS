#ifndef GMASTER_CV_2022_ARMORDETECTOR_H
#define GMASTER_CV_2022_ARMORDETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <inference_engine.hpp>
#include <iostream>
#include <chrono>
#include <cmath>
#include <Checksum.h>

#include "../../kalman/include/Kalmanfilter.h"
#include "../../kalman/include/guardpredict.h"
#include "../../comm/CommPort.h"
#include "../../windmill/include/windmill_run.h"
#include "../../camera/include/camera/CamWrapperDH.h"

#include "ArmorYOLO.h"

using namespace std;
using namespace InferenceEngine;

extern std::chrono::steady_clock::time_point e1;

struct TxPacket {
    unsigned char cache[packet_size];

    unsigned char &operator[](int p) {
        return cache[p];
    }

    unsigned char operator[](int p) const {
        return cache[p];
    }

    unsigned char *operator&() {
        return cache;
    }

    /*Constructor*/
    TxPacket() {
        memset(cache, 0, sizeof(0));
    }
};

class ArmorDetector
{
private:
    typedef struct {
        float prob;
        std::string name;
        cv::Rect2f rect;
    } Object;

    //have
    Detection_package package_get;
    //sol
    Kalmanfilter kalman;

    float offset_x;
    float offset_y;

    CommPort &comm;
    TxPacket packet;
    ArmorDetector::Object final_obj;
    ArmorDetector::Object last_obj;

public:

    int cnt = 0;
    //get
    Robotstatus robotstatus;
    //send
    RobotCMD robotcmd;

    std::shared_ptr<BUFF> buff;

    ArmorDetector(CommPort &c);

    ~ArmorDetector();

    void load_param(Robotstatus &status);

    void sendArmor();

    void sendData();

    void run(cv::Mat &);

    void cmd_start();

};

#endif //GMASTER_CV_2022_ARMORDETECTOR_H
