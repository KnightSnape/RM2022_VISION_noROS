#pragma once
#include"windmill_detect.h" 
#include"windmill_predict.h"



class BUFF
{
    public:
        typedef std::shared_ptr<BUFF> Ptr;
        BUFF();
        void set_start_time(std::chrono::steady_clock::time_point t);
        void load_params();
        bool run(Robotstatus &status,RobotCMD &cmd,cv::Mat img_src);
        void checkTarget(int sign_cnt);
        float getTargetPolarAngle(cv::Point2f center,cv::Point2f center_R);
        void getTargetTime();
        void getPredictPoint(cv::Point2f target_point);
        void getAimPoint(cv::Point2f target_point_);
        void buildRotatedRect(cv::Point2f target_point,cv::Point2f center);
        Point2f rotate(cv::Point2f target_point,cv::Point2f center,float theta_offset,float get_length);
        bool isSwitch(buff_kpt::Object object1,buff_kpt::Object object2);

        std::chrono::steady_clock::time_point start_t;
        std::chrono::steady_clock::time_point last_timestamp;


        buff_kpt yolo;
        std::vector<buff_kpt::Object> objects;
        Buff_Predictor predictor;
        
    private:
        bool is_last_target_exists;
        int lost_cnt;
        double last_target_area;
        double last_bullet_speed;
        Point2i last_roi_center;
        Point2i roi_offset;
        Size2d input_size;
        std::vector<buff_kpt::Object> last_objects;
        std::deque<buff_kpt::Object> final_object_deque;

        const int max_lost_cnt = 4;//最大丢失目标帧数
        const int max_v = 4;       //最大旋转速度(rad/s)
        const int max_delta_t = 100; //使用同一预测器的最大时间间隔(ms)
        const double fan_length = 0.7; //大符臂长(R字中心至装甲板中心)
        const double no_crop_thres = 2e-3;      //禁用ROI裁剪的装甲板占图像面积最大面积比值

        const int max_deque_length = 5;

        float last_pitch;
        float last_yaw;
 
        int sign_cnt;
        bool is_Start;
        int sign;
        int sign_rotate;
        int right_cnt;
        int left_cnt;
        int final_sign;

        Point2f now_center;
        Point2f now_center_R;

	double bullet_speed;

        inline double X(Point A,Point B)
        {
            return A.y*B.x - B.y*A.x;
        }

	inline int velocity_offset(double k1,double k2)
	{
	     return (int)(k1 * bullet_speed + k2);
	}

        inline bool IsTarget(int label)
        {
            return (label % 2 == 0);
        }

        template<typename T>
        inline T getDis(T a,T b)
        {
            return sqrt(a*a+b*b);
        }

        double rangedAngleRad(double &angle) {
        if (fabs(angle) >= CV_PI) {
            angle -= (angle / fabs(angle)) * CV_2PI;
            angle = rangedAngleRad(angle);
        }
        return angle;
    }

        
        


};

