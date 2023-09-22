#ifndef CVEKF
#define CVEKF

#include"GlobalParams.h"

class EKF_CV {
public:
    explicit EKF_CV(const Matrix51d &X0 = Matrix51d::Zero()) :
            Xe(X0), P(Matrix55d::Identity()), Q(Matrix55d::Identity()), R(Matrix33d::Identity()) {}

    void init(const Matrix51d &X0 = Matrix51d::Zero()) {
        Xe = X0;
    }

    template<class Func>
    Matrix51d predict(Func &&func) {
        ceres::Jet<double, 5> Xe_auto_jet[5];
        for (int i = 0; i < 5; i++) {
            Xe_auto_jet[i].a = Xe[i];
            Xe_auto_jet[i].v[i] = 1;
        }
        ceres::Jet<double, 5> Xp_auto_jet[5];
        func(Xe_auto_jet, Xp_auto_jet);
        for (int i = 0; i < 5; i++) {
            Xp[i] = Xp_auto_jet[i].a;
            J_F.block(i, 0, 1, 5) = Xp_auto_jet[i].v.transpose();
            P = J_F * P * J_F.transpose() + Q;
        }
        return Xp;
    }

    template<class Func>
    Matrix51d update(Func &&func, const Matrix31d &Y) {
        ceres::Jet<double, 5> Xp_auto_jet[5];
        for (int i = 0; i < 5; i++) {
            Xp_auto_jet[i].a = Xe[i];
            Xp_auto_jet[i].v[i] = 1;
        }
        ceres::Jet<double, 5> Yp_auto_jet[3];
        func(Xp_auto_jet, Yp_auto_jet);
        for (int i = 0; i < 3; i++) {
            Yp[i] = Yp_auto_jet[i].a;
            J_H.block(i, 0, 1, 5) = Yp_auto_jet[i].v.transpose();
        }
        K = P * J_H.transpose() * (J_H * P * J_H.transpose() + R).inverse();
        Xe = Xp + K * (Y - Yp);
        P = (Matrix55d::Identity() - K * J_H) * P;
        return Xe;
    }

    Matrix51d Xe;
    Matrix51d Xp;
    Matrix55d J_F;
    Matrix35d J_H;
    Matrix55d P;
    Matrix55d Q;
    Matrix33d R;
    Matrix53d K;
    Matrix31d Yp;
};


struct PredictFunc {
    template<class T>
    void operator()(const T x0[5], T x1[5]) {
        x1[0] = x0[0] + delta_t * x0[1];
        x1[1] = x0[1];
        x1[2] = x0[2] + delta_t * x0[3];
        x1[3] = x0[3];
        x1[4] = x0[4];
    }

    double delta_t;
};

template<class T>
void xyz2pyd(T xyz[3], T pyd[3]) {
    pyd[0] = ceres::atan2(xyz[2], ceres::sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1]));
    pyd[1] = ceres::atan2(xyz[1], xyz[0]);
    pyd[2] = ceres::sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]);
}

struct Measure {
    template<class T>
    void operator()(const T x[5], T y[3]) {
        T x_[3] = {x[0], x[2], x[4]};
        xyz2pyd(x_, y);
    }
};

#endif