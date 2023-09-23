//
// Created by whoismz on 1/1/22.
//

#include "../include/Kalmanfilter.h"

bool areacmp(const armor_detect_yolo::Object &a,const armor_detect_yolo::Object &b)
{
    return a.rect.area() > b.rect.area();
}

extern std::chrono::steady_clock::time_point e1;
Kalmanfilter::Kalmanfilter()
{

    // 30
    F_MAT = (cv::Mat_<double>(3, 3)
            << 1364.533879, 0.000000, 324.296573,
            0.000000, 1364.747010, 255.291946,
            0.000000, 0.000000, 1.000000);

    // k1 k2 p1 p2 k3
    C_MAT = (cv::Mat_<double>(1, 5)
            << -0.089238, -0.140301, 0.000000, 0.000000, 4.196557);
    R_CI_MAT = (cv::Mat_<double>(3, 3)
            << -0.12264121, 0.9921558, -0.02420736,
            -0.02062435, -0.0269341, -0.99942443,
            -0.99223675, -0.12207136, 0.0237658);


    cv::cv2eigen(R_CI_MAT, R_CI);
    cv::cv2eigen(F_MAT, F);
    cv::cv2eigen(C_MAT, C);

    Kalman::Matrix_61d X;
    Kalman::Matrix_66d A;
    Kalman::Matrix_66d P;
    Kalman::Matrix_66d Q;
    Kalman::Matrix_63d K;
    Kalman::Matrix_36d H;
    Kalman::Matrix_33d R;

    X << 0, 0, 0, 0, 0, 0;
    A <<
      1, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1;

    H <<
      1, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0;

    Q <<
      1000, 0, 0, 0, 0, 0,
            0, 1000, 0, 0, 0, 0,
            0, 0, 1000, 0, 0, 0,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1;

    R << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    kalman = Kalman(A, H, Q, R, X, 0);
    last_pitch = 0;
    last_yaw = 0;

    curr_vx = 0;
    curr_vy = 0;
    curr_vz = 0;

    bullet_speed = 16;
}

void Kalmanfilter::initParams(bool update_all)
{
    std::string path = "../config/ekf_predict.yaml";
    YAML::Node Config = YAML::LoadFile(path);

    ekf.Q(0,0) = Config["Q00"].as<float>();
    ekf.Q(1,1) = Config["Q11"].as<float>();
    ekf.Q(2,2) = Config["Q22"].as<float>();
    ekf.Q(3,3) = Config["Q33"].as<float>();
    ekf.Q(4,4) = Config["Q44"].as<float>();

    ekf.R(0,0) = Config["R00"].as<float>();
    ekf.R(1,1) = Config["R11"].as<float>();
    ekf.R(2,2) = Config["R22"].as<float>();

    if(update_all && !is_antitop)
    {
        ekf.Q(0,0) = Config["Q00_AC"].as<float>();
        ekf.Q(1,1) = Config["Q11_AC"].as<float>();
        ekf.Q(2,2) = Config["Q22_AC"].as<float>();
        ekf.Q(3,3) = Config["Q33_AC"].as<float>();
        ekf.Q(4,4) = Config["Q44_AC"].as<float>();

        ekf.R(0.0) = Config["R00"].as<float>();
        ekf.R(1,1) = Config["R11"].as<float>();
        ekf.R(2,2) = Config["R22"].as<float>();
    }

    if(is_antitop)
    {
        ekf.Q(0,0) = Config["Q00_ANTI"].as<float>();
        ekf.Q(1,1) = Config["Q11_ANTI"].as<float>();
        ekf.Q(2,2) = Config["Q22_ANTI"].as<float>();
        ekf.Q(3,3) = Config["Q33_ANTI"].as<float>();
        ekf.Q(4,4) = Config["Q44_ANTI"].as<float>();

        ekf.R(0,0) = Config["R00_ANTI"].as<float>();
        ekf.R(1,1) = Config["R11_ANTI"].as<float>();
        ekf.R(2,2) = Config["R22_ANTI"].as<float>();

        if(update_all)
        {
            ekf.Q(0,0) = Config["Q00_ANTI_AC"].as<float>();
            ekf.Q(1,1) = Config["Q11_ANTI_AC"].as<float>();
            ekf.Q(2,2) = Config["Q22_ANTI_AC"].as<float>();
            ekf.Q(3,3) = Config["Q33_ANTI_AC"].as<float>();
            ekf.Q(4,4) = Config["Q44_ANTI_AC"].as<float>();

            ekf.R(0,0) = Config["R00_ANTI_AC"].as<float>();
            ekf.R(1,1) = Config["R11_ANTI_AC"].as<float>();
            ekf.R(2,2) = Config["R22_ANTI_AC"].as<float>();
        }
    }

}

bool Kalmanfilter::checkBigArmor(Armor armor,float radio)
{
    if(armor.id == 0 || armor.id == 7)
        return true;
    if(((armor.id >= 2 && armor.id <= 4) || (armor.id >= 9 && armor.id <= 11)) && radio < armor_radio_threshold)
        return true;
    return false;
}

int Kalmanfilter::chooseTarget(const vector<Armor> &armors,int timestamp)
{
    //策略：近处英雄>近中距离大残>连续跟到目标>多个目标中面积最大的
    bool is_last_id_exists = false;
    int target_id;
    for(auto armor:armors)
    {
        if(armor.id == 1 && armor.center3d_cam.norm() <= hero_danger_zone)
        {
            return armor.id;
        }

        else if (armor.id == last_armor.id && abs(armor.area - last_armor.area) / (float) armor.area < 0.3 &&
                   abs(timestamp - prev_timestamp) < 30) {
            is_last_id_exists = true;
            target_id = armor.id;
        }
    }
    if(is_last_id_exists == true)
        return target_id;
    return (*armors.begin()).id;
}

cv::Point2f Kalmanfilter::getCenter(cv::Point2f pts[4])
{
    for (int i = 0; i < 4; ++i)
    {
        for (int j = i + 1; j < 4; ++j)
        {
            if (pts[i] == pts[j])
            {
                std::cout << "[Error] Unable to calculate center_R point." << std::endl;
                return cv::Point2f{0, 0};
            }
        }
    }
    cv::Point2f center(0, 0);
    if (pts[0].x == pts[2].x && pts[1].x == pts[3].x) {
        std::cout << "[Error] Unable to calculate center_R point." << std::endl;
    } else if (pts[0].x == pts[2].x && pts[1].x != pts[3].x) {
        center.x = pts[0].x;
        center.y = (pts[3].y - pts[1].y) / (pts[3].x - pts[1].x) * (pts[0].x - pts[3].x) + pts[3].y;
    } else if (pts[1].x == pts[3].x && pts[0].x != pts[2].x) {
        center.x = pts[1].x;
        center.y = (pts[2].y - pts[0].y) / (pts[2].x - pts[0].x) * (pts[1].x - pts[0].x) + pts[0].y;
    } else {
        center.x = (((pts[3].y - pts[1].y) / (pts[3].x - pts[1].x) * pts[3].x - pts[3].y + \
                    pts[0].y - (pts[2].y - pts[0].y) / (pts[2].x - pts[0].x) * pts[0].x)) / \
                    ((pts[3].y - pts[1].y) / (pts[3].x - pts[1].x) - (pts[2].y - pts[0].y) / (pts[2].x - pts[0].x));
        center.y = (pts[2].y - pts[0].y) / (pts[2].x - pts[0].x) * (center.x - pts[0].x) + pts[0].y;
    }
    return center;
}

bool Kalmanfilter::predict(RobotCMD& send, cv::Mat& im_show, Robotstatus& getdata)
{
    //start for detection
    Object.clear();
    auto t0 = std::chrono::steady_clock::now();
    vector<Armor> armors;
    bullet_speed = getdata.robot_speed;
    trajectory.getSpeed(getdata.robot_speed);
    //Object_new = kpt.work(im_show);
    //if(Object_new.empty())
    //{
    //    std::cout<<"No object"<<std::endl;
   //     is_last_target_exists = false;
    //    send.target_id = (uint8_t)0;
    //    return false;
    //}

    if(!yolo.work(im_show,Object))
    {
        std::cout<<"No object"<<std::endl;
        is_last_target_exists = false;
        send.target_id = (uint8_t)0;
        return false;
    }

    Eigen::Quaternionf q_raw(getdata.q[0],getdata.q[1],getdata.q[2],getdata.q[3]);
    Eigen::Quaternionf q(q_raw.matrix().transpose());
    Matrix33d R_IW = q.matrix().cast<double>();

    sort(Object.begin(),Object.end(), areacmp);
    if(Object.size() > armor_max_cnt)
        Object.resize(armor_max_cnt);

    for(auto object:Object)
    {
        Armor armor;
        armor.id = get_id(object.label);
        armor.color = get_color(object.label);
        armor.conf = object.prob;
        if(armor.color == (int)ColorChoose::BLUE)
        {
            armor.key = "B" + to_string(armor.id);
        }
        else
        {
            armor.key = "R" + to_string(armor.id);
        }
        bool Is_correct_point = true;
        for(int i=0;i<4;i++)
        {
            if(object.point[i].x == 0 || object.point[i].y == 0)
            {
                Is_correct_point = false;
                break;
            }
        }
        if(!Is_correct_point)
        {
            continue;
        }
        for(int i = 0;i < 4;i++)
        {
            armor.apex2d[i] = object.point[i];
        }
        Point2f apex_sum;
        for (auto apex: armor.apex2d)
            apex_sum += apex;
        armor.center2d = apex_sum / 4.f;
        //生成装甲板旋转矩形和ROI
        std::vector<Point2f> points_pic(armor.apex2d, armor.apex2d + 4);

        armor.roi = armor.rect;

        RotatedRect points_pic_rrect = cv::minAreaRect(points_pic);
        armor.rrect = points_pic_rrect;
        auto bbox = points_pic_rrect.boundingRect();

        auto x = bbox.x - 0.5 * bbox.width * (armor_roi_expand_ratio_width - 1);
        auto y = bbox.y - 0.5 * bbox.height * (armor_roi_expand_ratio_height - 1);

        auto apex_wh_ratio = max(points_pic_rrect.size.height, points_pic_rrect.size.width) /
                             min(points_pic_rrect.size.height, points_pic_rrect.size.width);
        int pnp_method;
        //装甲板少的时候使用 ITRATIVE, 多的时候使用 IPPE
        if (Object.size() <= 2)
            pnp_method = SOLVEPNP_ITERATIVE;
        else
            pnp_method = SOLVEPNP_IPPE;

        PnP_Target_Type type;
        if(checkBigArmor(armor,apex_wh_ratio))
            type = PnP_Target_Type::BIG_;
        else
            type = PnP_Target_Type::SMALL_;
        pnp_method = cv::SOLVEPNP_AP3P;
        auto result = PnP_get_pc(points_pic,type);
        if((isnan(result[0]))||(isnan(result[1]))||(isnan(result[2])))
            continue;

        armor.type = type;
        armor.center3d_cam = result;
        armor.area = object.rect.area();
        armor.center3d_world = pc_to_pw(result,R_IW);
        armors.push_back(armor);
    }

    if(armors.empty())
    {
        is_last_target_exists = false;
        std::cout<<"pnp failed"<<std::endl;
        send.target_id = (uint8_t)0;
        return false;
    }

    auto target_id = chooseTarget(armors,getdata.timestamp);

    string target_key;
    if(getdata.color == ColorChoose::RED)
        target_key = "B" + to_string(target_id);
    else     
        target_key = "R" + to_string(target_id);

    auto t3 = std::chrono::steady_clock::now();

    Armor target;
    Eigen::Vector3d aiming_point;
    std::vector<Armor> final_armors;

    bool found_target = false;

    for(auto armor:armors)
    {
        if(target_id != armor.id)
            continue;
        if((int)getdata.color != armor.color)
            continue;
        //和之前一样
        if(target_id == last_armor.id || last_armor.roi.contains((target.center2d)))
        {
            target = armor;
            found_target = true;
            is_target_switched = false;
            break;
        }
        if((target_id != last_armor.id || !last_armor.roi.contains((target.center2d))) && is_last_target_exists) 
        {
            target = armor;
            is_target_switched = true;
            found_target = true;
            break;
        }
        if(!is_last_target_exists)
        {
            target = armor;
            found_target = true;
            is_target_switched = true;
        }
    }

    if(!found_target)
    {
        is_last_target_exists = false;
        is_target_switched = true;
        std::cout<<"Not correct target"<<std::endl;
        return false;
    }

    Predict predictfunc;
    Measure measure;
    Antitop_Delay delay;
//    std::cout<<target.center3d_world.transpose()<<std::endl;
    re_project_point(im_show,target.center3d_world,R_IW,cv::Scalar(255,255,0));
//    std::cout<<bullet_speed<<std::endl;

//    if(is_target_switched)
//        std::cout<<"BRRRRRRRRRRRRRRRR"<<std::endl;

    if(target_id == 6)
	is_antitop = true;
    else
	is_antitop = false;
    float velocity = 0.0;
    if(!is_antitop)
    {
        if(is_target_switched)
        {
            initParams(false);
            Eigen::Matrix<double,5,1> Xr;
            Xr << target.center3d_world[0],0,target.center3d_world[1],0,target.center3d_world[2];
            ekf.init(Xr);
        }
        else
        {
            Eigen::Matrix<double, 5, 1> Xr;
            Xr << target.center3d_world[0],0,target.center3d_world[1],0,target.center3d_world[2];
            Eigen::Matrix<double, 3, 1> Yr;
            measure(Xr.data(), Yr.data());// 转化成相机求坐标系 Yr
            predictfunc.delta_t = Delta_T / 1e3;//设置距离上次预测的时间
            ekf.predict(predictfunc);
            Eigen::Matrix<double,5,1> Xe = ekf.update(measure,Yr); 
            double predict_time = target.center3d_world.norm() / bullet_speed / 2 + shoot_delay - 0.01;
            predictfunc.delta_t = predict_time;
            Eigen::Matrix<double,5,1> Xp;
            predictfunc(Xe.data(),Xp.data());

            Eigen::Vector3d c_pw{Xe(0,0),Xe(2,0),Xe(4,0)};
            target.center3d_world[0] = Xp(0,0);
            target.center3d_world[1] = Xp(2,0);
            target.center3d_world[2] = Xp(4,0);
        }
    }
    else
    {
        if(is_target_switched)
        {
            initParams(false);
            Eigen::Matrix<double,5,1> Xr;
            Xr << target.center3d_world[0],0,target.center3d_world[1],0,target.center3d_world[2];
            ekf.init(Xr);
        }
        else
        {
            Eigen::Matrix<double, 5, 1> Xr;
            Xr << target.center3d_world[0],0,target.center3d_world[1],0,target.center3d_world[2];
            Eigen::Matrix<double, 3, 1> Yr;
            measure(Xr.data(), Yr.data());// 转化成相机求坐标系 Yr
            predictfunc.delta_t = Delta_T / 1e3;//设置距离上次预测的时间
            ekf.predict(predictfunc);
            Eigen::Matrix<double,5,1> Xe = ekf.update(measure,Yr);
            //Xe(1,0) = -Xe(1,0);
	    //Xe(3,0) = -Xe(3,0);

	    if(Xe(1.0) > 0.1)
                Xe(1.0) = 0.1;
            if(Xe(1.0) < -0.1)
                Xe(1,0) = -0.1;
            if(Xe(3,0) > 0.1)
                Xe(3,0) = 0.1;
            if(Xe(3,0) < -0.1)
                Xe(3,0) = -0.1;
            double predict_time = target.center3d_world.norm() / bullet_speed / 3.2 + shoot_delay - 0.018;
            predictfunc.delta_t = predict_time;
            Eigen::Matrix<double,5,1> Xp;
            predictfunc(Xe.data(),Xp.data());

            Eigen::Vector3d c_pw{Xe(0,0),Xe(2,0),Xe(4,0)};
            target.center3d_world[0] = Xp(0,0);
            target.center3d_world[1] = Xp(2,0);
            target.center3d_world[2] = Xp(4,0);
        }
    }
//    std::cout<<velocity<<std::endl;

    //target.center3d_world = pc_to_pw(target.center3d_cam,R_IW);
    //re_project_point(im_show,target.center3d_world,R_IW,cv::Scalar(0,255,255));
    auto target_center_cam = pw_to_pc(target.center3d_world,R_IW);
    if(is_antitop)
    target.center3d_cam << target_center_cam[0],target.center3d_cam[1],target_center_cam[2];
    else
    target.center3d_cam << target_center_cam[0],target_center_cam[1],target_center_cam[2];

    target.center3d_world = pc_to_pw(target.center3d_cam,R_IW);
    re_project_point(im_show,target.center3d_world,R_IW,cv::Scalar(0,255,0));
//    std::cout<<target.center3d_world.transpose()<<std::endl;
//   std::cout<<target.center3d_cam.transpose()<<std::endl;
    last_armor = target;
    is_last_target_exists = true;
    double s_yaw = atan(-target.center3d_cam(0, 0) / target.center3d_cam(2, 0));
    double s_pitch = atan(-target.center3d_cam(1, 0) / target.center3d_cam(2, 0));

//    std::cout<<bullet_speed<<std::endl;

//    cv::imshow("1",im_show);
//    cv::waitKey(1);

// //    cout<<getdata.yaw<<" "<<getdata.pitch<<endl;
    send.yaw_angle = (float)(s_yaw);
    send.pitch_angle = (float)(s_pitch)+trajectory.TrajectoryCal(target.center3d_cam);
//     //cout<<s_yaw<<endl;
//     //send

    if (std::abs(send.yaw_angle) < 90 && std::abs(send.pitch_angle) < 90)
         send.shoot_mode = static_cast<uint8_t>(ShootMode::COMMON);
    else
         send.shoot_mode = static_cast<uint8_t>(ShootMode::CRUISE);

    initParams(true);
    send.priority = Priority::DANGER;

    send.target_id = static_cast<uint8_t>(target.id);

    auto t5 = std::chrono::steady_clock::now();
    auto delta_t_1 = std::chrono::duration<double,std::milli>(t5-t0).count();

    Delta_T = delta_t_1;

    return true;
}

void Kalmanfilter::frame_diff()
{
    double dt;
    unsigned long frames_size = frames_info.size();

    int frames_cnt = 30;

    if (frames_size < 5) {
        return;
    }

    if (frames_size < frames_cnt + 1)
    {
        dt = frames_info.back().timestamp - frames_info.front().timestamp;
        curr_vx = (frames_info.back().x - frames_info.front().x) * 100.0 / ((int) frames_size - 1);
        curr_vy = (frames_info.back().y - frames_info.front().y) * 100.0 / ((int) frames_size - 1);
        curr_vz = (frames_info.back().z - frames_info.front().z) * 100.0 / ((int) frames_size - 1);
        
    }
    else
    {
        dt = frames_info[frames_size - 1].timestamp - frames_info[frames_size - frames_cnt - 1].timestamp;
        curr_vx = (frames_info.back().x - frames_info[frames_size - frames_cnt - 1].x) * 100.0 / frames_cnt;
        curr_vy = (frames_info.back().y - frames_info[frames_size - frames_cnt - 1].y) * 100.0 / frames_cnt;
        curr_vz = (frames_info.back().z - frames_info[frames_size - frames_cnt - 1].z) * 100.0 / frames_cnt;
    }
}

Eigen::Vector3d Kalmanfilter::PnP_get_pc(const std::vector<Point2f> &p, PnP_Target_Type armor_number)
{
    static const std::vector<cv::Point3d> pw_small =
    {
            {-armor_small_l , -armor_small_w, 0},
            {-armor_small_l , armor_small_w, 0},
            {armor_small_l , -armor_small_w, 0},
            {armor_small_l , armor_small_w, 0}
    };

    static const std::vector<cv::Point3d> pw_big =
    {
           {-armor_big_l,-armor_big_w,0},
           {-armor_big_l,armor_big_w,0},
           {armor_big_l,-armor_big_w,0},
           {armor_big_l,armor_big_w,0}
    };

    cv::Mat rvec, tvec;
    if (armor_number)
    {
        cv::solvePnP(pw_big, p, F_MAT, C_MAT, rvec, tvec,false,cv::SOLVEPNP_AP3P );
    }

    else
        cv::solvePnP(pw_small, p, F_MAT, C_MAT, rvec, tvec,false,cv::SOLVEPNP_AP3P);

    Eigen::Vector3d pc;
    cv::cv2eigen(tvec, pc);

    pc << pc(0, 0) + sol_pc_x, pc(1, 0) + sol_pc_y, pc(2, 0) + sol_pc_z;

    return pc;
}


