#ifndef TRAJECTORY
#define TRAJECTORY

#include"../../others/GlobalParams.h"

using namespace std;
using namespace cv;


struct PnPInfo {
    Eigen::Vector3d armor_cam;
    Eigen::Vector3d armor_world;
    Eigen::Vector3d R_cam;
    Eigen::Vector3d R_world;
    Eigen::Vector3d euler;
    Eigen::Matrix3d rmat;
};

class CoordSolver {

public:
    CoordSolver();

    ~CoordSolver();

    void getSpeed(double speed);

    void load_param(string coord_path, string param_name);

    Eigen::Vector3d pc_to_pw(const Eigen::Vector3d &pc, const Eigen::Matrix3d &R_IW);

    Eigen::Vector3d pw_to_pc(const Eigen::Vector3d &pw, const Eigen::Matrix3d &R_IW);

    Eigen::Vector3d pc_to_pw2(const Eigen::Vector3d &pc, const Eigen::Matrix3d &R_IW);

    Eigen::Vector3d pw_to_pc2(const Eigen::Vector3d &pw, const Eigen::Matrix3d &R_IW);

    Eigen::Vector3d pc_to_pu(const Eigen::Vector3d &pc);

    PnPInfo pnp(const std::vector<Point2f> &points_pic, const Vision &vision, Eigen::Matrix3d &R_IW);

    cv::Point2f re_project_point(const Eigen::Vector3d &pw, const Eigen::Matrix3d &R_IW);

    Eigen::Vector2d getAngle(Eigen::Vector3d &xyz_cam, Eigen::Matrix3d &rmat);

    double TrajectoryCal(Matrix31d &xyz);

    Eigen::Vector3d rotationMatrixToEulerAngles(Matrix33d &R);

    cv::Mat F_Mat;

    cv::Mat C_Mat;

    Matrix33d F;

    Eigen::Matrix<double, 1, 5> C;

    cv::Mat R_CI_Mat;

    Eigen::Matrix4d R_CI;

    Eigen::Matrix4d R_IC;

    Eigen::Vector3d T_IW;

    Eigen::Matrix3d R_CI_NEW;

private:

    Matrix31d xyz_offset;

    Matrix21d angle_offset;

    int max_iter;

    float stop_error;

    int R_K_iter;

    double bullet_speed;


    const double k = 0.0196;

    const double g = 9.781;

    inline Eigen::Vector3d StaticPosOffset(const Eigen::Vector3d &xyz) {
        return xyz + xyz_offset;
    }

    inline Eigen::Vector2d StaticAngleOffset(const Eigen::Vector2d &angle) {
        return angle + angle_offset;
    }

    inline double CalPitch(const Eigen::Vector3d &xyz) {
        return -(atan2(xyz[1], sqrt(xyz[0] * xyz[0] + xyz[2] * xyz[2]) * 180 / PI));
    }

    inline double CalYaw(const Eigen::Vector3d &xyz) {
        return atan2(xyz[0], xyz[2]) * 180 / PI;
    }

    inline Eigen::Vector2d CalPitchYaw(const Eigen::Vector3d &xyz) {
        Eigen::Vector2d angle;
        angle << CalYaw(xyz), CalPitch(xyz);
        return angle;
    }
};

#endif