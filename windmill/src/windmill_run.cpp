#include"windmill_run.h"

BUFF::BUFF()
{
    yolo.load_params("../config/win-0703.xml");
    lost_cnt = 0;
    is_last_target_exists = false;
    input_size = {640,384};
    last_bullet_speed = 0;
    right_cnt = 0;
    left_cnt = 0;
    final_sign = 0;
    sign_cnt = 0;
    is_Start = false;
    bullet_speed = 26.0;
}

void BUFF::set_start_time(std::chrono::steady_clock::time_point t)
{
    this->start_t = t;
}

bool BUFF::run(Robotstatus &status,RobotCMD &cmd,cv::Mat img_src)
{
    sign_rotate = 0;
    auto time_start = std::chrono::steady_clock::now();
    //cout<<"speed:"<<bullet_speed<<endl;
    objects.clear();
    objects = yolo.work(img_src);
    if(objects.empty())
    {
        lost_cnt++;
        is_last_target_exists = false;
        last_target_area = 0;
        cmd.target_id = 255;
        return false;
    }
    if(status.robot_speed > 25.0 && status.robot_speed < 30.0)
	bullet_speed = status.robot_speed;
    Eigen::Quaternionf q_raw(status.q[0],status.q[1],status.q[2],status.q[3]);
    Eigen::Quaternionf q(q_raw.matrix().transpose());
    Matrix33d R_IW = q.matrix().cast<double>();

    auto time_infer = std::chrono::steady_clock::now();

    double time_length = std::chrono::duration<double,std::milli>(time_infer - start_t).count();
    double delta_time = std::chrono::duration<double,std::milli>(time_start - start_t).count();

    if(final_object_deque.size() < max_deque_length)
    {
        final_object_deque.push_back(objects[0]);
        auto time_end = std::chrono::steady_clock::now();
        last_timestamp = time_end;
        last_objects = objects;
        cmd.target_id = 255;
        return false;
    }

    else if(final_object_deque.size() == max_deque_length)
    {
        final_object_deque.pop_front();
        final_object_deque.push_back(objects[0]);
    }

    if(isSwitch(final_object_deque.front(),final_object_deque.back()))
    {
        final_object_deque.clear();
        if(sign == 0)
        {
            auto time_end = std::chrono::steady_clock::now();
            last_timestamp = time_end;
            last_objects = objects;
            cmd.target_id = 255;
            return false;
        }
    }
    if(is_Start == false)
    {
        is_Start = true;
        auto time_end = std::chrono::steady_clock::now();
        last_timestamp = time_end;
        last_objects = objects;
        cmd.target_id = 255;
        return false;
    }
    else
    {
        sign_cnt ++;
        checkTarget(sign_cnt);
    }    

    auto time_speed = std::chrono::steady_clock::now();

    double delta_t_speed = std::chrono::duration<double,std::milli>(time_speed - last_timestamp).count();

    double delta_t_speed_mul = 35;
    //cout<<delta_t_speed<<endl;
    float rotate_speed = 0;
    float Rotate_Speed = 0;

    for(int i=0;i<final_object_deque.size();i++)
    {
        float target_Angle = getTargetPolarAngle(final_object_deque.front().center,final_object_deque.front().center_R);
        float target_Angle_last = getTargetPolarAngle(final_object_deque.back().center,final_object_deque.back().center_R);
        auto delta_angle = target_Angle - target_Angle_last;
        Rotate_Speed += (delta_angle / delta_t_speed_mul)*1e3;
    }

    Rotate_Speed /= final_object_deque.size();
    for(int i=0;i<objects.size();i++)
    {
        float target_Angle = getTargetPolarAngle(objects[i].center,objects[i].center_R);
        float target_Angle_last = getTargetPolarAngle(last_objects[i].center,last_objects[i].center_R);
        //cout<<objects[i].center<<" "<<objects[i].center_R<<endl;
        //cout<<last_objects[i].center<<" "<<last_objects[i].center_R<<endl;
        //cout<<target_Angle<<" "<<target_Angle_last<<endl;
        auto delta_angle = target_Angle - target_Angle_last;
        //cout<<delta_angle<<endl;
        rotate_speed += (delta_angle / delta_t_speed)*1e3;
    }
    rotate_speed /= objects.size();

    if(isnan(rotate_speed))
    {
        last_timestamp = time_infer;
        last_objects = objects;
        cmd.target_id = 255;
        return false;
    }

    if(status.vision == Vision::WINDSMALL)
        predictor.mode = 0;
    else if(status.vision == Vision::WINDBIG)
        predictor.mode = 1;
    //std::cout<<sign<<std::endl;
    double theta_offset = 0;
    //predict
    if(!predictor.predict(rotate_speed,7.4,status.timestamp,theta_offset))
    {
        last_timestamp = time_infer;
        last_objects = objects;
        cmd.target_id = 255;
        return false;
    }
    float get_length;
    if(predictor.mode == 0)
	sign_rotate = sign;
    else
	sign_rotate = 1;

    Point2f hit_center,hit_center_final;
    float hit_center_angle;
    hit_center_final = rotate(objects[0].center,objects[0].center_R,theta_offset,get_length);

    Point2f Rect_Armor[5];
    vector<Point2f> Predict_Armor;

    std::cout<<theta_offset<<std::endl;
    std::cout<<objects[0].kpt[2]<<std::endl;

    Rect_Armor[0] = rotate(objects[0].kpt[0],objects[0].center_R,theta_offset,get_length);
    Rect_Armor[1] = rotate(objects[0].kpt[1],objects[0].center_R,theta_offset,get_length);
    Rect_Armor[2] = objects[0].center_R;
    Rect_Armor[3] = rotate(objects[0].kpt[3],objects[0].center_R,theta_offset,get_length);
    Rect_Armor[4] = rotate(objects[0].kpt[4],objects[0].center_R,theta_offset,get_length);

    //std::cout<<hit_center_final<<std::endl;
    circle(img_src,hit_center_final,2,Scalar(255,255,255),10);

    //circle(img_src,Rect_Armor[0],2,Scalar(255,0,255),10);
    //circle(img_src,Rect_Armor[1],2,Scalar(255,0,255),10);
    //circle(img_src,Rect_Armor[3],2,Scalar(255,0,255),10);
    //circle(img_src,Rect_Armor[4],2,Scalar(255,0,255),10);

    cv::imshow("dst",img_src);
    cv::waitKey(1);
    for(int i=0;i<5;i++)
    {
        Predict_Armor.push_back(Rect_Armor[i]);
    }
    int pitch_offset = velocity_offset(0.33,0);

    double dx = hit_center_final.x - 320 - 20;
    double dy = hit_center_final.y - 192 + 58 + pitch_offset;

    auto s_yaw = - dx / 1347.121380;
    auto s_pitch = - dy / 1348.912814;

    cmd.pitch_angle = s_pitch;
    cmd.yaw_angle = s_yaw;
    cmd.target_id = (uint8_t)1;
    cmd.priority = Priority::DANGER;

    auto time_end = std::chrono::steady_clock::now();
    last_timestamp = time_end;
    last_objects = objects;

    return true;
}

void BUFF::checkTarget(int sign_cnt)
{

    if(sign_cnt == 10)
    {
        now_center = objects[0].center;
        now_center_R = objects[0].center_R;
        return;
    }
    if(sign_cnt == 35 && X(objects[0].center - objects[0].center_R,now_center - now_center_R) > 0)
    {
        sign = 1;
        return;
    }
    else if(sign_cnt == 35 && X(objects[0].center - objects[0].center_R,now_center - now_center_R) < 0)
    {
        sign = -1;
        return;
    }
}

float BUFF::getTargetPolarAngle(cv::Point2f center,cv::Point2f center_R)
{
    return (atan2(center.y - center_R.y,center.x - center_R.x));
}

Point2f BUFF::rotate(cv::Point2f target_point,cv::Point2f center,float theta_offset,float get_length)
{
    get_length = getDis(target_point.x - center.x,target_point.y - center.y);
    cv::Point2f hit_center_final;
    float hit_center_angle;
    hit_center_angle = (getTargetPolarAngle(target_point,center)+theta_offset*sign_rotate);
    hit_center_final = Point2f(cosf(hit_center_angle)*get_length+center.x,sinf(hit_center_angle)*get_length+center.y);
    return hit_center_final;
}
void BUFF::buildRotatedRect(cv::Point2f target_point,cv::Point2f center)
{

}

bool BUFF::isSwitch(buff_kpt::Object object1,buff_kpt::Object object2)
{
    auto angle1 = getTargetPolarAngle(object1.center,object1.center_R);
    auto angle2 = getTargetPolarAngle(object2.center,object2.center_R);
    auto delta_angle = (angle1 - angle2) * 180 / PI;
    if(abs(delta_angle) > 30)
    {
        return true;
    }
    else
    {
        return false;
    }
}
