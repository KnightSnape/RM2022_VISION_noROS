#include "ArmorDetector.h"

std::chrono::steady_clock::time_point e1;

ArmorDetector::ArmorDetector(CommPort &c) : comm(c) 
{
    std::string path = "../config/config.yaml";
    YAML::Node Config = YAML::LoadFile(path);

    int state = Config["test"]["state"].as<int>();
    if(state == 0)
    {
        offset_x = Config["test"]["low_speed_offset_x"].as<float>();
        offset_y = Config["test"]["low_speed_offset_y"].as<float>();
    }
    else 
    {
        offset_x = Config["test"]["high_speed_offset_x"].as<float>();
        offset_y = Config["test"]["high_speed_offset_y"].as<float>();
    }
}

ArmorDetector::~ArmorDetector() = default;

void ArmorDetector::run(cv::Mat &img_src) {
    comm.Turnstatus(robotstatus);
    cmd_start();
    std::chrono::steady_clock::time_point e2 = std::chrono::steady_clock::now();
    double final = std::chrono::duration<double, std::milli>(e2 - e1).count();
    robotstatus.timestamp = (int) (final);
    bool have = kalman.predict(robotcmd, img_src, robotstatus);
//    if (FOR_PC)
//        robotstatus.vision == Vision::CLASSIC;
//    if (robotstatus.vision == Vision::CLASSIC || robotstatus.vision == Vision::SENTLY) {
//        bool have = kalman.predict(robotcmd, img_src, robotstatus);
//    } else if (robotstatus.vision == Vision::WINDSMALL || robotstatus.vision == Vision::WINDBIG) {
//        bool have = buff.run(img_src, robotstatus, robotcmd);
//    } else {
//        cout << "[Error]Vision is wrong" << endl;
//        exit(-1);
//    }
    sendArmor();
}

void ArmorDetector::sendArmor() {
    std::thread t(&ArmorDetector::sendData, this);
    t.detach();
}

void ArmorDetector::sendData() {
    packet[0] = 0x5A;
    if (robotcmd.priority == Priority::CORE) packet[1] = 0;
    if (robotcmd.priority == Priority::DANGER) packet[1] = 1;
    packet[2] = robotcmd.target_id;
    union {
        float actual;
        uint8_t raw[4];
    } tx_x{}, tx_y{};

    tx_x.actual = robotstatus.pitch + (float) robotcmd.pitch_angle + offset_x;
    tx_y.actual = robotstatus.yaw + (float) robotcmd.yaw_angle + offset_y;

    for (int i = 0; i < 4; i++) {
        packet[2 + i] = tx_x.raw[i];  // x
        packet[6 + i] = tx_y.raw[i];  // y
    }
    Crc8Append(&packet, 12);
    //printf("  pitch %.2f yaw %.2f id %d cmd %d  time %.3f  \n", robotcmd.pitch_angle, robotcmd.yaw_angle, robotcmd.target_id, packet[1], robotstatus.timestamp);

    comm.Write(&packet, 12, true);
}

void ArmorDetector::cmd_start() {
    robotcmd.priority = Priority::CORE;
    robotcmd.target_id = 255;
    robotcmd.pitch_angle = 0;
    robotcmd.yaw_angle = 0;
    package_get.detection.clear();
}
