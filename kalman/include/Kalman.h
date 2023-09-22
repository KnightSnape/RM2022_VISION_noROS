#ifndef GMASTER_WM_NEW_KALMAN_H
#define GMASTER_WM_NEW_KALMAN_H

#include <Eigen/Dense>
#include"Trajectory.h"

#include "../../others/GlobalParams.h"
//TODO FINISH
class Kalman
{
    public:
        using Matrix_31d = Eigen::Matrix<double, 3, 1>;
        using Matrix_33d = Eigen::Matrix<double, 3, 3>;
        using Matrix_36d = Eigen::Matrix<double, 3, 6>;
        using Matrix_61d = Eigen::Matrix<double, 6, 1>;
        using Matrix_63d = Eigen::Matrix<double, 6, 3>;
        using Matrix_66d = Eigen::Matrix<double, 6, 6>;

        double last_vx = 0, last_vy = 0, last_vz = 0;
    private:
        Matrix_61d x_k1;
        Matrix_63d K;
        Matrix_66d A;
        Matrix_36d H;
        Matrix_33d R;
        Matrix_66d Q;
        Matrix_66d P;
        double last_t{0};
    public:
    Kalman() = default;

    ~Kalman() = default;

    Kalman(Matrix_66d A, Matrix_36d H, Matrix_66d Q, Matrix_33d R, Matrix_61d x_init, double t) {
        reset(std::move(A), std::move(H), std::move(Q), std::move(R), std::move(x_init), t);
    }

    void reset(Matrix_66d A_, Matrix_36d H_, Matrix_66d Q_, Matrix_33d R_, Matrix_61d x_init, double t) {
        this->A = std::move(A_);
        this->H = std::move(H_);
        this->P = Matrix_66d::Zero();
        this->R = std::move(R_);
        this->Q = std::move(Q_);
        x_k1 = std::move(x_init);
        last_t = t;
    }

    void reset(Matrix_61d x_init, double t) {
        x_k1 = std::move(x_init);
        last_t = t;
    }

    void reset(double x, double t) {
        x_k1(0, 0) = x;
        last_t = t;
    }

    Matrix_61d update(const Matrix_31d &z_k, double t) {

        auto dt = t - last_t;
        last_t = t;

        // R is Q, Q is R, are you ok?
        // 设置转移矩阵中的时间项
        A(0, 3) = dt;
        A(1, 4) = dt;
        A(2, 5) = dt;

        x_k1[3] = last_vx;
        x_k1[4] = last_vy;
        x_k1[5] = last_vz;

        // 预测下一时刻的值
        Matrix_61d x_k_k1 = A * x_k1;

        // x 的先验估计由上一个时间点的后验估计值和输入信息给出

        // 求协方差
        P = A * P * A.transpose() + Q;
        // 计算先验均方差 p(n|n-1)=A^2*p(n-1|n-1)+q

        // 计算kalman增益
        K = P * H.transpose() * (H * P * H.transpose() + R).inverse();
        // Kg(k)= P(k|k-1) H’ / (H P(k|k-1) H’ + R)

        // 修正结果，即计算滤波值
        x_k1 = x_k_k1 + K * (z_k - H * x_k_k1);
        // 利用残余的信息改善对 x(t) 的估计，给出后验估计，这个值也就是输出  X(k|k)= X(k|k-1)+Kg(k) (Z(k)-H X(k|k-1))

        // 更新后验估计
        P = (Matrix_66d::Identity() - K * H) * P;
        // 计算后验均方差  P[n|n]=(1-K[n]*H)*P[n|n-1]

//        printf("%f %f %f\n", x_k_k1[3], x_k_k1[4], x_k_k1[5]);

        return x_k1;
    }

};

#endif //GMASTER_WM_NEW_KALMAN_H
