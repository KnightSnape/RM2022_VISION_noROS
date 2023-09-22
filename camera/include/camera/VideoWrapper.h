//
// Created by whoismz on 1/2/22.
//

#ifndef GMASTER_WM_NEW_VIDEOWRAPPER_H
#define GMASTER_WM_NEW_VIDEOWRAPPER_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "WrapperHead.h"


class VideoWrapper:public WrapperHead {
public:
    VideoWrapper(const std::string& filename);
    ~VideoWrapper();


    /**
     * @brief initialize cameras
     * @return bool value: whether it success
     */
    bool init() final;


    /**
     * @brief read images from camera
     * @param src_left : output source video of left camera
     * @param src_right : output source video of right camera
     * @return bool value: whether the reading is successful
     */
    bool read(cv::Mat &src) final;
private:
    cv::VideoCapture video;

};

#endif //GMASTER_WM_NEW_VIDEOWRAPPER_H
