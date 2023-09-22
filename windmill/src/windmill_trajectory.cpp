#include"windmill_trajectory.h"

CoordSolver::CoordSolver() {
    string path = "../config/config.yaml";

    string param_name = "test";//now

    load_param(path, param_name);
}

CoordSolver::~CoordSolver() {

}

void CoordSolver::getSpeed(double speed) {
    bullet_speed = speed;
}

void CoordSolver::load_param(string coord_path, string param_name) {
    YAML::Node Config = YAML::LoadFile(coord_path);

    Eigen::MatrixXd mat_xyz_offset(1, 3);
    Eigen::MatrixXd mat_angle_offset(1, 2);
    Eigen::MatrixXd mat_c(1, 5);
    Eigen::MatrixXd mat_f(3, 3);
    Eigen::MatrixXd mat_r_ci(4, 4);
    Eigen::MatrixXd mat_r_ic(4, 4);
    Eigen::MatrixXd mat_t_iw(1, 3);

    max_iter = Config[param_name]["max_iter"].as<int>();
    stop_error = Config[param_name]["stop_error"].as<float>();
    R_K_iter = Config[param_name]["R_K_iter"].as<int>();

    auto read_vector = Config[param_name]["Intrinsic"].as<vector<float>>();
    initMatrix(mat_f, read_vector);
    F = mat_f;
    cv::eigen2cv(F, F_Mat);

    read_vector = Config[param_name]["Coeff"].as<vector<float>>();
    initMatrix(mat_c, read_vector);
    C = mat_c;
    cv::eigen2cv(C, C_Mat);

    read_vector = Config[param_name]["R_CI"].as<vector<float>>();
    initMatrix(mat_r_ci, read_vector);
    //R_CI = mat_r_ci;
    R_CI = Eigen::Matrix4d::Identity();
    cv::eigen2cv(R_CI, R_CI_Mat);

    read_vector = Config[param_name]["R_IC"].as<vector<float>>();
    initMatrix(mat_r_ic, read_vector);
    R_IC = Eigen::Matrix4d::Identity();
    //R_IC = mat_r_ic;

    read_vector = Config[param_name]["T_CI"].as<vector<float>>();
    initMatrix(mat_t_iw, read_vector);
    T_IW = mat_t_iw.transpose();

    read_vector = Config[param_name]["xyz_offset"].as<vector<float>>();
    initMatrix(mat_xyz_offset, read_vector);
    xyz_offset = mat_xyz_offset.transpose();

    read_vector = Config[param_name]["angle_offset"].as<vector<float>>();
    initMatrix(mat_angle_offset, read_vector);
    angle_offset = mat_angle_offset.transpose();

}

Eigen::Vector2d CoordSolver::getAngle(Eigen::Vector3d &xyz_cam, Eigen::Matrix3d &rmat) {
    auto xyz_offseted = StaticPosOffset(xyz_cam);
    auto xyz_world = pc_to_pw(xyz_offseted, rmat);
    auto angle_cam = CalPitchYaw(xyz_cam);

    auto pitch_offset = TrajectoryCal(xyz_world);

    angle_cam[1] = angle_cam[1] + pitch_offset;
    auto angle_offseted = StaticAngleOffset(angle_cam);
    return angle_offseted;
}

double CoordSolver::TrajectoryCal(Matrix31d &xyz) {
    auto dist_vertical = xyz[2];
    auto vertical_tmp = dist_vertical;
    auto dist_horizonal = sqrt(xyz.squaredNorm() - dist_vertical * dist_vertical);

    auto pitch = atan(dist_vertical / dist_horizonal) * 180 / PI;
    auto pitch_new = pitch;

    auto pitch_offset = 0.0;

    for (int i = 0; i < max_iter; i++) {
        auto x = 0.0;
        auto y = 0.0;
        auto p = tan(pitch_new / 180 * PI);
        auto v = bullet_speed;
        auto u = v / sqrt(1 + pow(p, 2));
        auto delta_x = dist_horizonal / R_K_iter;
        for (int j = 0; j < R_K_iter; j++) {
            auto k1_u = -k * u * sqrt(1 + pow(p, 2));
            auto k1_p = -g / pow(u, 2);
            auto k1_u_sum = u + k1_u * (delta_x / 2);
            auto k1_p_sum = p + k1_p * (delta_x / 2);

            auto k2_u = -k * k1_u_sum * sqrt(1 + pow(k1_p_sum, 2));
            auto k2_p = -g / pow(k1_u_sum, 2);
            auto k2_u_sum = u + k2_u * (delta_x / 2);
            auto k2_p_sum = p + k2_p * (delta_x / 2);

            auto k3_u = -k * k2_u_sum * sqrt(1 + pow(k2_p_sum, 2));
            auto k3_p = -g / pow(k2_u_sum, 2);
            auto k3_u_sum = u + k3_u * (delta_x / 2);
            auto k3_p_sum = p + k3_p * (delta_x / 2);

            auto k4_u = -k * k3_u_sum * sqrt(1 + pow(k3_p_sum, 2));
            auto k4_p = -g / pow(k3_u_sum, 2);

            u += (delta_x / 6) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u);
            p += (delta_x / 6) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p);

            x += delta_x;
            y += p * delta_x;
        }
        auto error = dist_vertical - y;
        if (abs(error) <= stop_error) {
            break;
        } else {
            vertical_tmp += error;
            pitch_new = atan(vertical_tmp / dist_horizonal) * 180 / PI;
        }

    }
    return pitch_new - pitch;
}

Eigen::Vector3d CoordSolver::pc_to_pw(const Eigen::Vector3d &pc, const Eigen::Matrix3d &R_IW) {
    Eigen::Vector4d point_camera_tmp;
    Eigen::Vector4d point_imu_tmp;
    Eigen::Vector3d point_imu;
    Eigen::Vector3d point_world;
    Eigen::Vector3d pw_new;

    point_camera_tmp << pc[0], pc[1], pc[2], 1;
    point_imu_tmp = R_IC * point_camera_tmp;
    point_imu << point_camera_tmp[0], point_camera_tmp[1], point_camera_tmp[2];
    point_imu -= T_IW;
    Eigen::Vector3d pw = R_IW * point_imu;
    pw_new << pw(0, 0), pw(2, 0), -pw(1, 0);
    return pw;

}

Eigen::Vector3d CoordSolver::pw_to_pc(const Eigen::Vector3d &pw, const Eigen::Matrix3d &R_IW) {
    Eigen::Vector4d point_camera_tmp;
    Eigen::Vector4d point_imu_tmp;
    Eigen::Vector3d point_imu;
    Eigen::Vector3d point_camera;
    Eigen::Vector3d pw_new;

    //cout<<pw.transpose()<<endl;
    pw_new << pw(0, 0), -pw(2, 0), pw(1, 0);
    //cout<<pw_new.transpose()<<endl;
    point_imu = R_IW.transpose() * pw;
    point_imu += T_IW;
    point_imu_tmp << point_imu[0], point_imu[1], point_imu[2], 1;
    point_camera_tmp = R_CI * point_imu_tmp;
    point_camera << point_camera_tmp[0], point_camera_tmp[1], point_camera_tmp[2];
    return point_camera;
}

Eigen::Vector3d CoordSolver::pc_to_pu(const Eigen::Vector3d &pc) {
    return F * pc / pc(2, 0);
}

cv::Point2f CoordSolver::re_project_point(const Eigen::Vector3d &pw, const Eigen::Matrix3d &R_IW) {
    auto pc = pw_to_pc(pw, R_IW);
    auto pu = pc_to_pu(pc);
    cv::Point2f final_point;
    final_point = cv::Point2f(pu(0, 0), pu(1, 0));
    return final_point;
}

PnPInfo CoordSolver::pnp(const std::vector<Point2f> &points_pic, const Vision &vision, Eigen::Matrix3d &R_IW) {

    static const std::vector<cv::Point3d> pw_wind =
            {
                    {-wind_big_l, wind_big_w,  0},
                    {-wind_big_l, -wind_big_w, 0},
                    {0,           -0.7,        -0.05},
                    {wind_big_l,  -wind_big_w, 0},
                    {wind_big_l,  wind_big_w,  0}
            };

    Eigen::Matrix3d rmat_eigen;
    Eigen::Vector3d R_center_world = {0, -0.7, -0.05};
    Eigen::Vector3d tvec_eigen;
    Eigen::Vector3d coord_camera;

    cv::Mat rvec, tvec, rmat;
    solvePnP(pw_wind, points_pic, F_Mat, C_Mat, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);

    PnPInfo result;
    Eigen::Vector3d pc;
    Rodrigues(rvec, rmat);
    cv::cv2eigen(rmat, rmat_eigen);
    cv::cv2eigen(tvec, tvec_eigen);


    result.armor_cam = tvec_eigen;
    result.armor_world = pc_to_pw(result.armor_cam, R_IW);
    result.R_cam = (rmat_eigen * R_center_world) + tvec_eigen;
    result.R_world = pc_to_pw(result.R_cam, R_IW);
    Eigen::Matrix3d rmat_eigen_world = R_IW * (R_IC.block(0, 0, 3, 3) * rmat_eigen);
    // result.euler = rotationMatrixToEulerAngles(rmat_eigen_world);
    result.euler = rotationMatrixToEulerAngles(rmat_eigen_world);
    result.rmat = rmat_eigen_world;
    //complete
    return result;

}

Eigen::Vector3d CoordSolver::rotationMatrixToEulerAngles(Matrix33d &R) {
    double sy = sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));
    bool singular = sy < 1e-6;
    double x, y, z;
    if (!singular) {
        x = atan2(R(2, 1), R(2, 2));
        y = atan2(-R(2, 0), sy);
        z = atan2(R(1, 0), R(0, 0));
    } else {
        x = atan2(-R(1, 2), R(1, 1));
        y = atan2(-R(2, 0), sy);
        z = 0;
    }
    return {z, y, x};
}

Eigen::Vector3d CoordSolver::pc_to_pw2(const Eigen::Vector3d &pc, const Eigen::Matrix3d &R_IW) {
    auto R_WC = (R_CI_NEW * R_IW).transpose();
    return R_WC * pc;

}

Eigen::Vector3d CoordSolver::pw_to_pc2(const Eigen::Vector3d &pw, const Eigen::Matrix3d &R_IW) {
    auto R_CW = R_CI_NEW * R_IW;
    return R_CW * pw;
}