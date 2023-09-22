#ifndef RUN
#define RUN

#include"windmill_detect.h"
#include"windmill_trajectory.h"
#include"windmill_predict.h"
#include"../../particle/ParticleFilter.h"
#include"fan_tracker.h"

using namespace std;
using namespace cv;

class BUFF {
public:
    BUFF();

    ~BUFF();

    bool run(cv::Mat &src, Robotstatus &robotstatus, RobotCMD &robotcmd);

private:
    const string network_path = "../config/buff.xml";
    const string camera_param_path = "../config/config.yaml";
    bool is_last_target_exists;
    int lost_cnt;
    int last_timestamp;
    double last_target_area;
    double last_bullet_speed;
    Point2i last_roi_center;
    Point2i roi_offset;
    Size2d input_size;
    std::vector<FanTracker> trackers;      //tracker
    const int max_lost_cnt = 4;//最大丢失目标帧数
    const int max_v = 4;       //最大旋转速度(rad/s)
    const int max_delta_t = 100; //使用同一预测器的最大时间间隔(ms)
    const double fan_length = 0.7; //大符臂长(R字中心至装甲板中心)
    const double no_crop_thres = 2e-3;      //禁用ROI裁剪的装甲板占图像面积最大面积比值

    cv::Point2f center_R;
    cv::Point2f center_R_old;
    cv::Point2f target_point;
    cv::Point2f predict_point;

    float current_pitch;
    float current_yaw;
    float target_polar_angle;
    float last_target_polar_angle_judge_change;
    float last_target_polar_angle_predict_time;
    float last_target_polar_angle_time_point;
    float predict_time;
    float predict_time_cnt;
    float predict_time_sum;
    double predict_rad;
    double extra_delta_x, extra_delta_y;
    double yaw_rotation, pitch_rotation;

    Fan last_fan;


    BuffDetector detector;
    BuffPredictor predictor;
    CoordSolver coordsolver;

    bool chooseTarget(vector<Fan> &fans, Fan &target);

    Point2i cropImageByROI(Mat &img);

    void changeTarget();

    void getTargetPolarAngle();

    void getTargetTime();

    void rotate(cv::Point2f target_point);

    void getPredictPoint(cv::Point2f target_point);

    void getAimPoint(cv::Point2f target_point_);

    void judgeShoot();

};

#endif