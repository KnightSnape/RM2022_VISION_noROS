#include"windmill_predict.h"

Buff_Predictor::Buff_Predictor() {
    is_params_confirmed = false;
    params[0] = 0;
    params[1] = 0;
    params[2] = 0;
    params[3] = 0;

    YAML::Node config = YAML::LoadFile(pf_path);
    pf_param_loader.initParam(config, "buff");
}

Buff_Predictor::~Buff_Predictor() {

}

bool Buff_Predictor::predict(double speed, double dist, int timestamp, double &result) {
    auto t1 = std::chrono::steady_clock::now();
    TargetInfo target = {speed, dist, timestamp};
    if (mode != last_mode)//change
    {
        last_mode = mode;
        history_info.clear();
        pf.initParam(pf_param_loader);
        is_params_confirmed = false;
    }

    if (history_info.size() == 0 || target.timestamp - history_info.front().timestamp >= max_timespan) {
        history_info.clear();
        history_info.push_back(target);
        params[0] = 0;
        params[1] = 0;
        params[2] = 0;
        params[3] = 0;
        pf.initParam(pf_param_loader);
        last_target = target;
        is_params_confirmed = false;
        return false;
    }

    auto is_ready = pf.is_ready;
    Eigen::VectorXd measure(1);
    measure << speed;
    pf.update(measure);

    if (is_ready) {
        auto predict = pf.predict();
        target.speed = predict[0];
    }

    auto deque_len = 0;
    if (mode == 0) {
        deque_len = history_deque_len_uniform;
    } else if (mode == 1) {
        if (!is_params_confirmed)
            deque_len = history_deque_len_cos;
        else
            deque_len = history_deque_len_phase;
    }
    double rotate_speed_sum;
    int rotate_sign;
    double mean_velocity;
    if(mode == 1)
    {
        if (history_info.size() < deque_len) {
            history_info.push_back(target);
            last_target = target;
            return false;
        } else if (history_info.size() == deque_len) {
            history_info.pop_front();
            history_info.push_back(target);
        } else if (history_info.size() > deque_len) {
            while (history_info.size() >= deque_len) {
                history_info.pop_front();
            }
            history_info.push_back(target);
        }

        rotate_speed_sum = 0;
        rotate_sign;
        for (auto target_info: history_info) {
            rotate_speed_sum += target_info.speed;
        }
        mean_velocity = rotate_speed_sum / history_info.size();
    }


    if (mode == 0) {
        params[3] = 1.0;
    } else if (mode == 1) {
        if (!is_params_confirmed) {
            ceres::Problem problem;
            ceres::Solver::Options options;
            ceres::Solver::Summary summary;
            double params_fitting[4] = {1, 1, 1, mean_velocity};

            if (rotate_speed_sum / fabs(rotate_speed_sum) >= 0)
                rotate_sign = 1;
            else
                rotate_sign = -1;
            for (auto target_info: history_info) {
                problem.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 4>(
                                new CURVE_FITTING_COST((float) (target_info.timestamp) / 1e3,
                                                       target_info.speed * rotate_sign)
                        ),
                        new ceres::CauchyLoss(0.5),
                        params_fitting
                );
            }
            //should be fixed in the real place
            problem.SetParameterLowerBound(params_fitting, 0, 0.7);
            problem.SetParameterUpperBound(params_fitting, 0, 1.2);
            problem.SetParameterLowerBound(params_fitting, 1, 1.6);
            problem.SetParameterUpperBound(params_fitting, 1, 2.2);
            problem.SetParameterLowerBound(params_fitting, 2, -PI);
            problem.SetParameterUpperBound(params_fitting, 2, PI);
            problem.SetParameterLowerBound(params_fitting, 3, 0.5);
            problem.SetParameterUpperBound(params_fitting, 3, 2.5);

            ceres::Solve(options, &problem, &summary);
            double params_tmp[4] = {params_fitting[0] * rotate_sign, params_fitting[1], params_fitting[2],
                                    params_fitting[3] * rotate_sign};

            auto rmse = evalRMSE(params_tmp);
            if (rmse > max_rmse) {
                cout << summary.BriefReport() << endl;
                cout << "RMSE is too high, Fitting failed!" << endl;
                return false;
            } else {
                cout << "[BUFF_PREDICT]Fitting Succeed! RMSE: " << rmse << endl;
                params[0] = params_fitting[0] * rotate_sign;
                params[1] = params_fitting[1];
                params[2] = params_fitting[2];
                params[3] = params_fitting[3] * rotate_sign;
                is_params_confirmed = true;
            }
        } else {
            ceres::Problem problem;
            ceres::Solver::Options options;
            ceres::Solver::Summary summary;       // 优化信息
            double phase;

            for (auto target_info: history_info) {
                problem.AddResidualBlock(     // 向问题中添加误差项
                        // 使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
                        new ceres::AutoDiffCostFunction<CURVE_FITTING_COST_PHASE, 1, 1>(
                                new CURVE_FITTING_COST_PHASE((float) (target_info.timestamp) / 1e3,
                                                             (target_info.speed - params[3]) * rotate_sign, params[0],
                                                             params[1], params[3])
                        ),
                        new ceres::CauchyLoss(1e1),
                        &phase                 // 待估计参数
                );
            }

            //设置上下限
            problem.SetParameterLowerBound(&phase, 0, -CV_PI);
            problem.SetParameterUpperBound(&phase, 0, CV_PI);

            ceres::Solve(options, &problem, &summary);
            double params_new[4] = {params[0], params[1], phase, params[3]};
            auto old_rmse = evalRMSE(params);
            auto new_rmse = evalRMSE(params_new);
            if (new_rmse < old_rmse) {
                LOG(INFO) << "[BUFF_PREDICT]Params Updated! RMSE: " << new_rmse;
                params[2] = phase;
            }
            cout << "RMSE:" << new_rmse << endl;
        }
    }
    int delay = (mode == 1 ? delay_big : delay_small);

    float delta_time_estimate = ((double) dist / bullet_speed) * 1e3 + delay;

    float timespan = history_info.back().timestamp;

    float time_estimate = delta_time_estimate + timespan;

    result = calcAimingAngleOffset(params, timespan / 1e3, time_estimate / 1e3, mode);

    last_target = target;
    //TODO when zR can get plt,we can draw the graph

    return true;
}

bool Buff_Predictor::setBulletSpeed(double speed) {
    bullet_speed = speed;
    return true;
}

double Buff_Predictor::calcAimingAngleOffset(double params[4], double t0, double t1, int mode) {
    auto a = params[0];
    auto omega = params[1];
    auto theta = params[2];
    auto b = params[3];
    double theta1;
    double theta0;
    // cout<<"t1: "<<t1<<endl;
    // cout<<"t0: "<<t0<<endl;
    //f(x) = a * sin(ω * t + θ) + b
    //对目标函数进行积分
    if (mode == 0)//适用于小符模式
    {
        theta0 = b * t0;
        theta1 = b * t1;
    } else {
        theta0 = (b * t0 - (a / omega) * cos(omega * t0 + theta));
        theta1 = (b * t1 - (a / omega) * cos(omega * t1 + theta));
    }
    // cout<<(theta1 - theta0) * 180 / CV_PI<<endl;
    return theta1 - theta0;
}

double Buff_Predictor::evalRMSE(double params[4]) {
    double rmse_sum = 0;
    double rmse = 0;
    for (auto target_info: history_info) {
        auto time = (float) (target_info.timestamp) / 1e3;
        auto pred = params[0] * sin(params[1] * time + params[2]) + params[3];
        auto measure = target_info.speed;
        rmse_sum += pow((pred - measure), 2);
    }
    rmse = sqrt(rmse_sum / history_info.size());
    return rmse;

}
