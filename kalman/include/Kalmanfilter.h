#ifndef GMASTER_WM_NEW_KALMANFILTER_H
#define GMASTER_WM_NEW_KALMANFILTER_H

#include "Kalman.h"
#include "EKF.h"
#include<yaml-cpp/yaml.h>
#include "../../armor/include/ArmorYOLO.h"
#include "../../armor/include/ArmorNewYOLO.h"

using namespace cv;

enum TargetType {
    SMALL, BIG
};

//装甲板
struct Armor {
    int id;
    int color;
    int area;
    double conf;
    string key;
    Point2f apex2d[4];
    Rect rect;
    RotatedRect rrect;
    Rect roi;
    Point2f center2d;
    Eigen::Vector3d center3d_cam;
    Eigen::Vector3d center3d_world;
    Eigen::Vector3d euler;
    Eigen::Vector3d predict;

    PnP_Target_Type type;
};

class ArmorTracker
{
    public:

        ArmorTracker(Armor src, int src_timestamp)
        {
            last_armor = src;
            last_timestamp = src_timestamp;
            key = src.key;
            is_initialized = false;
            hit_score = 0;
            history_info.push_back(src);
            calcTargetScore();
        }

        bool calcTargetScore()
        {
            vector<Point2f> points;
            float rotate_angle;

            RotatedRect rotated_rect = last_armor.rrect;

            if (rotated_rect.size.width > rotated_rect.size.height)
                rotate_angle = rotated_rect.angle;
            else
                rotate_angle = 90 - rotated_rect.angle;

            hit_score = log(0.15 * (90 - rotate_angle) + 10) * (last_armor.area);
            return true;
        }

        bool update(Armor new_armor, int new_timestamp)
        {
            if (history_info.size() <= max_history_len)
            {
                history_info.push_back(new_armor);
            }
            else
            {
                history_info.pop_front();
                history_info.push_back(new_armor);
            }

            is_initialized = true;
            prev_armor = last_armor;
            prev_timestamp = last_timestamp;
            last_armor = new_armor;
            last_timestamp = new_timestamp;

            calcTargetScore();
            return true;
        }

        Armor prev_armor;                       //上一次装甲板
        Armor last_armor;                       //本次装甲板
        bool is_initialized;                    //是否完成初始化
        int last_selected_timestamp;            //该Tracker上次被选为目标tracker时间戳
        int prev_timestamp;                     //上次装甲板时间戳
        int last_timestamp;                     //本次装甲板时间戳
        int history_type_sum;                   //历史次数之和
        int selected_cnt;                       //该Tracker被选为目标tracker次数和
        const int max_history_len = 4;          //历史信息队列最大长度
        double hit_score;                       //该tracker可能作为目标的分数,由装甲板旋转角度,距离,面积大小决定
        double velocity;
        double radius;
        string key;

        std::deque<Armor> history_info;//目标队列
};

struct Predict {
    /*
     * 此处定义匀速直线运动模型
     */
    template<class T>
    void operator()(const T x0[5], T x1[5]) {
        x1[0] = x0[0] + delta_t * x0[1];  //0.1
        x1[1] = x0[1];  //100
        x1[2] = x0[2] + delta_t * x0[3];  //0.1
        x1[3] = x0[3];  //100
        x1[4] = x0[4];  //0.01
    }

    double delta_t;
};

template<class T>
void xyz2pyd(T xyz[3], T pyd[3]) {
    /*
     * 工具函数：将 xyz 转化为 pitch、yaw、distance
     */
    pyd[0] = ceres::atan2(xyz[2], ceres::sqrt(xyz[0]*xyz[0]+xyz[1]*xyz[1]));  // pitch
    pyd[1] = ceres::atan2(xyz[1], xyz[0]);  // yaw
    pyd[2] = ceres::sqrt(xyz[0]*xyz[0]+xyz[1]*xyz[1]+xyz[2]*xyz[2]);  // distance
}

struct Measure {
    /*
     * 工具函数的类封装
     */
    template<class T>
    void operator()(const T x[5], T y[3]) {
        T x_[3] = {x[0], x[2], x[4]};
        xyz2pyd(x_, y);
    }
};

class Kalmanfilter
{
public:
    void initParams(bool update_all);
    bool checkBigArmor(Armor armor,float radio);
    int chooseTarget(const vector<Armor> &armors,int timestamp);
    bool predict(RobotCMD& send, cv::Mat& im_show, Robotstatus& getdata);
    Eigen::Vector3d PnP_get_pc(const std::vector<Point2f> &p, PnP_Target_Type armor_number);
    void frame_diff();
    void get_speed();
    struct FramesInfo 
    {
        double x;
        double y;
        double z;
        double timestamp;
    };
    std::vector<FramesInfo> frames_info;
    struct ROI
    {
        bool ROI_selected = false;
        cv::Rect2f ROI_bbox;
        int last_class = -1;
        ROI() = default;
        ROI(cv::Rect2f&& bbox, int& last) :ROI_selected(true), ROI_bbox(bbox), last_class(last) {}
        inline void clear()
        {
            ROI_selected = false;
            last_class = -1;
        }
        ~ROI() = default;
    };


    Detection_package package_get;

    Kalmanfilter();

    ~Kalmanfilter() = default;

    cv::Point2f getCenter(cv::Point2f[4]);

    private:

    bool is_last_target_exists;
    
    Eigen::Matrix3d R_CI;           // 陀螺仪坐标系到相机坐标系旋转矩阵EIGEN-Matrix
    Eigen::Matrix3d F;              // 相机内参矩阵EIGEN-Matrix
    Eigen::Matrix<double, 1, 5> C;  // 相机畸变矩阵EIGEN-Matrix
    cv::Mat R_CI_MAT;               // 陀螺仪坐标系到相机坐标系旋转矩阵CV-Mat
    cv::Mat F_MAT;                  // 相机内参矩阵CV-Mat
    cv::Mat C_MAT;                  // 相机畸变矩阵CV-Mat

    armor_detect_yolo yolo;
    vector<armor_detect_yolo::Object> Object;
    AdaptiveEKF<5, 3> ekf;  // 创建ekf
    Kalman kalman;

    double test_t = 0;
    double curr_vx, curr_vy, curr_vz;
    ROI roi;

    int prev_timestamp;//上一帧时间戳
    bool is_target_switched;

    double last_pitch;
    double last_yaw;

    double Delta_T;

    Trajectory trajectory;
    float bullet_speed;

    Armor last_armor;

    inline Eigen::Vector3d pc_to_pw(const Eigen::Vector3d &pc, const Eigen::Matrix3d &R_IW) {
        auto R_WC = (R_CI * R_IW).transpose();
        return R_WC * pc;
    }

    inline Eigen::Vector3d pw_to_pc(const Eigen::Vector3d &pw, const Eigen::Matrix3d &R_IW) {
        auto R_CW = R_CI * R_IW;
        return R_CW * pw;
    }

    inline Eigen::Vector3d pc_to_pu(const Eigen::Vector3d &pc) {
        return F * pc / pc(2, 0);
    }

    inline void re_project_point(cv::Mat &image, const Eigen::Vector3d &pw,
                                 const Eigen::Matrix3d &R_IW, const cv::Scalar &color)
   {

        Eigen::Vector3d pc = pw_to_pc(pw, R_IW);
        Eigen::Vector3d pu = pc_to_pu(pc);

//        printf("Af: x: %f y: %f\n\n", pu(0, 0), pu(1, 0));
        cv::circle(image, {int(pu(0, 0)), int(pu(1, 0))}, 4, color, 4);
    }

    inline double getDis(double A, double B)
    {
        return sqrt(A * A + B * B);
    }

    inline double getDis(cv::Point2f A,cv::Point2f B)
    {
        return getDis(B.x-A.x,B.y-A.y);
    }

    inline int get_color(int label)
    {
        return (label < 7)?(1):(0);
    }
    inline int get_id(int label)
    {
        return (label < 7)?(label + 1):(label - 6);
    }

};

#endif //GMASTER_WM_NEW_KALMANFILTER_H
