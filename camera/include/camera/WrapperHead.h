//
// Created by whoismz on 1/2/22.
//

#ifndef GMASTER_WM_NEW_WRAPPERHEAD_H
#define GMASTER_WM_NEW_WRAPPERHEAD_H

#include <opencv2/core/core.hpp>

/**
 * @brief A virtual class for wrapper of camera and video files
 */
class WrapperHead {
public:
    virtual ~WrapperHead() = default;;
    virtual bool init() = 0;
    virtual bool read(cv::Mat &src) = 0;
};



#endif //GMASTER_WM_NEW_WRAPPERHEAD_H
