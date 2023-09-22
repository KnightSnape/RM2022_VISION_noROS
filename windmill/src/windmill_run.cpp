#include"windmill_run.h"

BUFF::BUFF()
{
    detector.initModel(network_path);
    coordsolver.load_param(camera_param_path,"test");
    lost_cnt = 0;
    is_last_target_exists = false;
    input_size = {640,384};
    last_bullet_speed = 0;
    //fmt::print(fmt::fg(fmt::color::pale_violet_red), "[BUFF] Buff init model success! Size: {} {}\n", input_size.height, input_size.width);
}
BUFF::~BUFF()
{

}
#ifdef USING_ROI
Point2i BUFF::cropImageByROI(Mat &img)
{
        if (!is_last_target_exists)
    {
        //当丢失目标帧数过多或lost_cnt为初值
        if (lost_cnt > max_lost_cnt || lost_cnt == 0)
        {
            return Point2i(0,0);
        }
    }
    //若目标大小大于阈值
    // cout<<last_target_area / img.size().area()<<endl;
    if ((last_target_area / img.size().area()) > no_crop_thres)
    {
        return Point2i(0,0);
    }
    //处理X越界
    if (last_roi_center.x <= input_size.width / 2)
        last_roi_center.x = input_size.width / 2;
    else if (last_roi_center.x > (img.size().width - input_size.width / 2))
        last_roi_center.x = img.size().width - input_size.width / 2;
    //处理Y越界
    if (last_roi_center.y <= input_size.height / 2)
        last_roi_center.y = input_size.height / 2;
    else if (last_roi_center.y > (img.size().height - input_size.height / 2))
        last_roi_center.y = img.size().height - input_size.height / 2;

    //左上角顶点
    auto offset = last_roi_center - Point2i(input_size.width / 2, input_size.height / 2);
    Rect roi_rect = Rect(offset, input_size);
    img(roi_rect).copyTo(img);

    return offset;
}
#endif 

bool BUFF::chooseTarget(vector<Fan> &fans, Fan &target) //目标选择函数
{
    float max_area = 0;
    int target_idx = 0;
    int target_fan_cnt = 0;
    for (auto fan : fans)
    {
        if (fan.id == 1)
        {
            target = fan;
            target_fan_cnt++;
        }
    }
    if (target_fan_cnt == 1)
        return true;
    else
        return false;
}

bool BUFF::run(cv::Mat &src,Robotstatus &robotstatus,RobotCMD &robotcmd) //能量机关主函数
{
    auto time_start = std::chrono::steady_clock::now();
    std::vector<BuffObject> objects;
    std::vector<Fan> fans;
    if(robotstatus.vision == Vision::CLASSIC)
    {
        cout<<"[Error]come to the wrong place"<<endl;
        return false;
    }

    auto q_ = robotstatus.q;
    Eigen::Quaternionf q_raw(q_[0], q_[1], q_[2], q_[3]);
    //Eigen::Matrix3d R_IW = q_raw.toRotationMatrix().cast<double>();
    Eigen::Matrix3d R_IW = Eigen::Matrix3d::Identity();
    //Eigen::Quaternionf q(q_raw.matrix().transpose());

    //Matrix33d R_IW = q.matrix().cast<double>();
//#ifdef USING_ROI
 //   roi_offset = cropImageByROI(src);
//#endif  //USING_ROI

    auto time_crop = std::chrono::steady_clock::now();
    imshow("windmill_inference",src);
    waitKey(1);
    if(!detector.detect(src,objects,robotstatus))
    {
        if(FOR_IMSHOW)
        {
            if(DEBUG)
            {
                line(src,Point2f(src.size().width / 2,0),Point2f(src.size().width / 2,src.size().height),Scalar(0,255,0),1);
                line(src,Point2f(0,src.size().height / 2),Point2f(src.size().width,src.size().height / 2),Scalar(0,255,0),1);
            }
            namedWindow("windmill_inference",0);
            imshow("windmill_inference",src);
            waitKey(1);
        }
        lost_cnt++;
        is_last_target_exists = false;
        last_target_area = 0;
        //send no target
        robotcmd.priority = Priority::CORE;
        robotcmd.target_id = 255;

        return false;
    }
    //return true;
    auto time_infer = std::chrono::steady_clock::now();
    if(DEBUG)
    {
        cout << src.cols << " " << src.rows << endl;
        cout << objects[0].rect.x << " " << objects[0].rect.y << endl;
        circle(src, Point2f(objects[0].rect.x, objects[0].rect.y), 2, Scalar(0, 255, 0), 1, 8, 1);
        imshow("windmill_inference", src);
        waitKey(1);
    }

    


    return true;
}

