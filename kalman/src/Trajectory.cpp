#include"Trajectory.h"

Trajectory::Trajectory()
{
    max_iter = 10;

    stop_error = 0.001;

    R_K_iter = 60;

    std::string path = "../config/config.yaml";
    YAML::Node Config = YAML::LoadFile(path);
    int state = Config["test"]["state"].as<int>();
    if(state == 0)
    {
        bullet_speed = Config["test"]["low_speed"].as<float>();
    }
    else
    {
        bullet_speed = Config["test"]["high_speed"].as<float>();
    }

}

double Trajectory::TrajectoryCal(Matrix31d &xyz)
{
    xyz[0] /= 100.0f;
    xyz[1] /= 100.0f;
    xyz[2] /= 100.0f;

    auto dist_vertical = xyz[1];
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
    return (pitch_new - pitch) / 180 * PI;
}

double Trajectory::getSpeed(double speed)
{
    return speed;
}


