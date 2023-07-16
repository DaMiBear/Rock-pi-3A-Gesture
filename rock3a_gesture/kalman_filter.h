#ifndef KALMANFILTER_H
#define KALMANFILTER_H

#include "yolo-fastestv2-anchorfree.h"
#include "NumCpp.hpp"
#include "opencv2/core/types.hpp"

class KalmanFilter
{
public:
    /* use xyxy bbox to init kalman filter */
    KalmanFilter(cv::Point &measurement);

    void predict();
    void update(cv::Point measurement);

    /* return xyxy after kalman filter */
    void get_state(cv::Point &inxy);

private:
    nc::NdArray<float> mean;        // x shape:(4,1)
    nc::NdArray<float> covariance;  // P shape:(4,4)
    nc::NdArray<float> _motion_mat; // Fk shape:(4,4)
    nc::NdArray<float> _update_mat; // Hk shape:(2,4)
    float _std_weight_position;
    float _std_weight_velocity;

    /* 计算中间变量 */
    void project(nc::NdArray<float> &project_mean,
                 nc::NdArray<float> &project_cov);
    /* cholesky分解, return L, A = L * L.T */
    nc::NdArray<float> cholesky_decomposition(nc::NdArray<float> &A);

    /* After cholesky_decomposition(), use L to solve Ax=b */
    nc::NdArray<float> cholesky_solve(nc::NdArray<double> &lower, nc::NdArray<float> b);
};

#endif // KALMANFILTER_H
