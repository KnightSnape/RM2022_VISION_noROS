#include "armor/include/ArmorDetector.h"
#include "camera/include/camera/CamWrapperDH.h"
#include "opencv2/core/core_c.h"
#include <signal.h>
#include <fstream>
using namespace cv;
using namespace std;

Mat img_src;
Camera *cam = nullptr;
ArmorDetector *adetector = nullptr;
CommPort comm;
TickMeter meter;
Robotstatus robotstatus;

bool linkCam_DaHeng();

void withVideo(const std::string &path);

std::string record_path()
{
    std::string path = "../test_armor_videos/";
    std::string add_path;
    int cnt = 1;
    add_path = path + std::to_string(cnt) + "_video.avi";
    while(true)
    {
	std::ifstream ifs(add_path);
        if(!ifs.is_open())
	{
	    path = add_path;
	    return path;
	}
        else
	{
            cnt++;
            add_path = path + std::to_string(cnt) + "_video.avi";
	}
    }

}

std::string record_path2()
{
    std::string path = "../test_armor_videos/";
    std::string add_path;
    int cnt = 1;
    add_path = path + std::to_string(cnt) + "_detect_video.avi";
    while(true)
    {
	std::ifstream ifs(add_path);
	if(!ifs.is_open())
	{
	    path = add_path;
	    return path;
	}
	else
	{
	    cnt++;
	    add_path = path + std::to_string(cnt) + "_detect_video.avi";
	}
    }
}

int main() {
    adetector = new ArmorDetector(comm);
    if (FOR_PC)
        withVideo("/home/zr/test_for_net.mp4");
    else
        linkCam_DaHeng();
    //    linkCam_MindVision(); //如果需要使用Midvison在使用
    return 0;
}

void withVideo(const std::string &path) {

    VideoCapture cap(path);
    while (true) {
        cap.read(img_src);
        if (img_src.empty()) {
            cout << "没有找到视频" << endl;
            break;
        }
        meter.start();
        adetector->run(img_src);
        meter.stop();
        printf("Time: %f\n", meter.getTimeMilli());
        meter.reset();
    }
    cap.release();
}

bool linkCam_DaHeng() 
{
    bool flag = false; // tradition
    comm.Start();
//    cam = new DHCamera("KF0190050040");
      cam = new DHCamera("KN0210060029");
    cam->init(0, 0, 640, 384, 13000, 8, false);
    std::string path;
    std::string path2;
    path = record_path();
    path2 = record_path2();
    cv::VideoWriter writer_1(path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 100, cv::Size(640, 384));
//    cv::VideoWriter writer_2(path2, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(640, 480));

    e1 = std::chrono::steady_clock::now();
    while (true) {
        if ((robotstatus.vision == Vision::WINDSMALL || robotstatus.vision == Vision::WINDBIG) && flag == false) {
            flag = true;
            cam->init(272, 400, 640, 384, 10000, 8, false);
        } else if ((robotstatus.vision == Vision::CLASSIC || robotstatus.vision == Vision::SENTLY) && flag == true) {
            flag = false;
            cam->init(272, 400, 640, 384, 10000, 8, false);
        }
        if (!cam->start())
            return false;
        cam->read(img_src);
        if (img_src.empty()) break;
        meter.start(); //计时器
        writer_1 << img_src;
        adetector->run(img_src);
        meter.stop(); //停止计时
        printf("Time1: %f\n", meter.getTimeMilli()); //打印时长
        meter.reset();
//        writer_2 << img_src;
    }
    return true;
}



