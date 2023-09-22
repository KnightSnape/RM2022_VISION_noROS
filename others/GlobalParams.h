//
// Created by whoismz on 1/2/22.
//

#ifndef GMASTER_WM_NEW_GLOBALPARAMS_H
#define GMASTER_WM_NEW_GLOBALPARAMS_H

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <ceres/ceres.h>
#include <thread>
#include <chrono>
#include<mutex>
#include<vector>
#include<string>
#include<stdint.h>
#include<iostream>
#include<cstdio>
#include <yaml-cpp/yaml.h>
#include <ctime>
#include <random>
#include"../windmill/include/Time.h"

using namespace std;

#define USING_ROI

using Matrix66d = Eigen::Matrix<double, 6, 6>;
using Matrix61d = Eigen::Matrix<double, 6, 1>;
using Matrix16d = Eigen::Matrix<double, 1, 6>;
using Matrix11d = Eigen::Matrix<double, 1, 1>;
using Matrix33d = Eigen::Matrix<double, 3, 3>;
using Matrix31d = Eigen::Matrix<double, 3, 1>;
using Matrix13d = Eigen::Matrix<double, 1, 3>;
using Matrix99d = Eigen::Matrix<double, 9, 9>;
using Matrix91d = Eigen::Matrix<double, 9, 1>;
using Matrix19d = Eigen::Matrix<double, 1, 9>;
using Matrix93d = Eigen::Matrix<double, 9, 3>;
using Matrix39d = Eigen::Matrix<double, 3, 9>;
using Matrix21d = Eigen::Matrix<double, 2, 1>;
using Matrix22d = Eigen::Matrix<double, 2, 2>;
using Matrix12d = Eigen::Matrix<double, 1, 2>;
using Matrix22f = Eigen::Matrix<float, 2, 2>;
using Matrix21f = Eigen::Matrix<float, 2, 1>;
using Matrix12f = Eigen::Matrix<float, 1, 2>;
using Matrix11f = Eigen::Matrix<float, 1, 1>;
using Matrix55d = Eigen::Matrix<double, 5, 5>;
using Matrix51d = Eigen::Matrix<double, 5, 1>;
using Matrix15d = Eigen::Matrix<double, 1, 5>;
using Matrix53d = Eigen::Matrix<double, 5, 3>;
using Matrix35d = Eigen::Matrix<double, 3, 5>;

//====================armor======================//

constexpr double armor_big_l = 0.1175;

constexpr double armor_big_w = 0.0625;

constexpr double armor_small_l = 0.069;

constexpr double armor_small_w = 0.0625;

constexpr double guard_pc_x = 0.0;

constexpr double guard_pc_y = 0.0;

constexpr double guard_pc_z = 0.0;

constexpr double sol_pc_x = 0.0;

constexpr double sol_pc_y = 0.0;

constexpr double sol_pc_z = 0.0;

constexpr int armor_max_cnt = 8;

constexpr double t = 0.06;

constexpr double shoot_delay = 0.11;

constexpr float height_thres = 4;

constexpr float high_thres = 0.6;

constexpr float low_thres = 0.4;

constexpr int important_kill = 1;

constexpr int killer_point = 100;

constexpr int killer_point_for_2 = 80;

constexpr double antitop_x_v_proportion = 0.;

constexpr double antitop_y_v_proportion = 0.;

constexpr double armor_roi_expand_ratio_width = 1;

constexpr double armor_roi_expand_ratio_height = 1.4;

constexpr float ac_x_v_coefficient = 0.5f;

constexpr float ac_y_v_coefficient = 0.5f;

constexpr double distance_max = 16;

constexpr double hero_danger_zone = 0.99;

constexpr float switch_armor_size_proportion = 1.1;

constexpr float armor_radio_threshold = 1.2;

constexpr double ac_init_min_age = 1;

constexpr bool DEBUG = false;
//you can use this to print something to DEBUG

constexpr bool FOR_IMSHOW = false;

constexpr bool FOR_SHOW_FANS = false;

constexpr bool FOR_PC = false;
//If you plan to run code on a computer without inter GPU, please change this option to true

#define PI 3.141592653589793238

const int INF = 0x7f7f7f7f;

constexpr size_t packet_size = 15;

//=======================Wind=========================//

constexpr double wind_small_l = 0.066;

constexpr double wind_small_w = 0.027;

constexpr double wind_big_l = 0.1125;

constexpr double wind_big_w = 0.027;

constexpr double energy_small_speed = 60;

constexpr double energy_delay_time = 320;

constexpr double energy_extra_delta_y = 0;

constexpr double energy_extra_delta_x = 0;

constexpr float RED_COMPENSATE_YAW = 0;

constexpr float RED_COMPENSATE_PITCH = 0;

constexpr float BLUE_COMPENSATE_YAW = 0;

constexpr float BLUE_COMPENSATE_PITCH = 0;

constexpr double MCU_DELTA_X = 0;

constexpr double MCU_DELTA_Y = 0;

constexpr double CLOCKWISE = 1;

enum class OURSELVES {
    GUARD = 0,
    OTHERS = 1,
};


enum class ColorChoose : uint8_t {
    RED = 0,
    BLUE = 1,
};

enum class GameState : uint8_t {
    SHOOT_NEAR_ONLY = 0,
    SHOOT_FAR = 1,
    COMMON = 255,
};

enum class Priority : uint8_t {
    CORE = 0,
    DANGER = 1,
    NONE = 255,
};

enum class ShootMode : uint8_t {
    COMMON = 0,
    DISTANT = 1,
    ANTITOP = 2,
    SWITCH = 4,
    FOLLOW = 8,
    CRUISE = 16,
    EXIST_HERO = 32,

};

enum class Vision : uint8_t {
    WINDSMALL = 0,
    WINDBIG = 1,
    SENTLY = 2,
    CLASSIC = 3,

};

enum PnP_Target_Type {SMALL_, BIG_, BUFF_, TAG_};

struct Robotstatus {
    std::array<double, 4> q;
    Vision vision = Vision::WINDSMALL;//uint8_t
    GameState gamestate = GameState::COMMON;
    ColorChoose color = ColorChoose::RED;
    uint8_t target_id = 255;
    float robot_speed = 14.;
    //blood
    std::array<uint16_t, 6> enemy;
    int timestamp = 0;
    float pitch;
    float yaw;
    uint8_t lrc = 0;

}__attribute__((packed));

struct RobotCMD {
    uint8_t start = (unsigned) 's';

    Priority priority; // uint_8
    uint8_t target_id = 255;//uint
    float pitch_angle = 0;
    float yaw_angle = 0;//place
    //float pitch_speed = 0;
    //float yaw_speed = 0;
    //float distance = 0;
    uint8_t shoot_mode = static_cast<uint8_t>(ShootMode::CRUISE);
    uint8_t lrc = 0;
    uint8_t end = (unsigned) 'e';

}__attribute__((packed));

struct bbox_t {
    cv::Rect rect;
    cv::Point2f pts[4];
    float confident;
    int color; //0->blue,1->red,2->grey
    int ID;//0 guard 1,2,3,4,5 6 base

    bool operator==(const bbox_t &bbox) const {
        return (rect == bbox.rect) && (pts == bbox.pts) && (confident == bbox.confident) && (color == bbox.color) &&
               (ID == bbox.ID);
    }

    bool operator!=(const bbox_t &bbox) const {
        return (rect != bbox.rect) || (pts != bbox.pts) || (confident != bbox.confident) || (color != bbox.color) ||
               (ID != bbox.ID);
    }
};

struct Detection_package {
    std::vector<bbox_t> detection;
    cv::Mat img;
    std::array<double, 4> q;
    double timestamp = 0;
};

constexpr uint8_t ourselves = static_cast<uint8_t>(OURSELVES::OTHERS);

template<typename T>
bool initMatrix(Eigen::MatrixXd &matrix, std::vector<T> &vector) {
    int cnt = 0;
    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.cols(); j++) {
            matrix(i, j) = vector[cnt];
            cnt++;
        }
    }
    return true;
}

#endif //GMASTER_WM_NEW_GLOBALPARAMS_H
