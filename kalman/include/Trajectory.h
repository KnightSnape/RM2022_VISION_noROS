#ifndef GMASTER_CV_2022_TRAJECTORY_H
#define GMASTER_CV_2022_TRAJECTORY_H

#include "../../others/GlobalParams.h"

class Trajectory {
public:

    Trajectory();

    ~Trajectory() = default;

    double TrajectoryCal(Matrix31d &xyz);

    double getSpeed(double speed);

private:
    float g = 9.786;

    float k = 0.0196;

    int max_iter;

    float stop_error;

    int R_K_iter;

    float bullet_speed = 18;                

};

#endif //GMASTER_CV_2022_TRAJECTORY_H
