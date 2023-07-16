#include "kalman_filter.h"

KalmanFilter::KalmanFilter(cv::Point &measurement_xy)
{
    /* note: Bk and uk is 0 */
    int ndim = 2;
    int dt = 1;
    _motion_mat = nc::eye<float>(2 * ndim); // Fk, shape:(4,4)
    for (int i = 0; i < ndim; i++)
        _motion_mat(i, ndim + i) = dt;                       // use "()" not "[]"
    _update_mat = nc::eye<float>(nc::Shape(ndim, 2 * ndim)); // shape:(2,4)
    _std_weight_position = 1. / 10;
    _std_weight_velocity = 1. / 160;

    /* initiate */
    nc::NdArray<float> measurement = {(float)(measurement_xy.x), (float)(measurement_xy.y)}; // shape:(1,2)
    nc::NdArray<float> mean_pos = measurement;
    nc::NdArray<float> mean_vel = nc::zeros_like<float>(mean_pos);

    mean = nc::column_stack({mean_pos, mean_vel}); // shape:(1,4)
    mean = mean.transpose();                       // xk-1|k-1, shape:(4,1):x, y, vx, vy
    nc::NdArray<float> std = {
        2 * _std_weight_position * measurement(0, 0),
        2 * _std_weight_position * measurement(0, 1),
        10 * _std_weight_velocity * measurement(0, 0),
        10 * _std_weight_velocity * measurement(0, 1)}; // shape:(4,)
    covariance = nc::diag(nc::square(std));             // Pk-1|k-1, shape:(4,4)
}

void KalmanFilter::predict()
{
    nc::NdArray<float> std_pos = {
        _std_weight_position * mean(0, 0),
        _std_weight_position * mean(1, 0)};
    nc::NdArray<float> std_vel = {
        _std_weight_velocity * mean(0, 0),
        _std_weight_velocity * mean(1, 0)};
    nc::NdArray<float> motion_cov = nc::diag(
        nc::square(nc::column_stack({std_pos, std_vel}))); // 系统噪声: Qk shape:(4,4)
    mean = nc::dot(_motion_mat, mean);                     // xk|k-1 = Fk * xk-1|k-1 + 0
    covariance = nc::linalg::multi_dot(
                     {_motion_mat, covariance, _motion_mat.transpose()}) +
                 motion_cov; // Pk|k-1 = Fk * Pk-1|k-1 * Fk.T + Qk
}

void KalmanFilter::project(nc::NdArray<float> &project_mean,
                           nc::NdArray<float> &project_cov)
{
    nc::NdArray<float> std = {
        10 * _std_weight_position * mean(0, 0),
        10 * _std_weight_position * mean(1, 0),
    };
    nc::NdArray<float> innovation_cov = nc::diag(nc::square(std)); // 测量噪声: Rk
    // Hk * xk|k-1
    project_mean = nc::dot(_update_mat, mean);
    // Sk = Hk * Pk|k-1 * Hk.T + Rk
    project_cov = nc::linalg::multi_dot(
                      {_update_mat, covariance, _update_mat.transpose()}) +
                  innovation_cov;
}

nc::NdArray<float> KalmanFilter::cholesky_decomposition(nc::NdArray<float> &A)
{
    nc::NdArray<float> lower = nc::zeros_like<float>(A);
    // Decomposing a matrix into Lower Triangular
    for (int i = 0; i < A.shape().rows; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            float sum = 0;

            if (j == i) // summation for diagonals
            {
                for (int k = 0; k < j; k++)
                    sum += pow(lower(j, k), 2);
                lower(j, j) = sqrt(A(j, j) - sum);
            }
            else
            {
                // Evaluating L(i, j) using L(j, j)
                for (int k = 0; k < j; k++)
                    sum += (lower(i, k) * lower(j, k));
                lower(i, j) = (A(i, j) - sum) / lower(j, j);
            }
        }
    }
    return lower;
}

nc::NdArray<float> KalmanFilter::cholesky_solve(nc::NdArray<double> &lower, nc::NdArray<float> b)
{
    if (b.shape().cols != 1)
    {
        printf("[ERROR] the shape of 'b' must be (n,1)\n");
        return nc::zeros_like<float>(b);
    }
    nc::NdArray<float> Y = nc::zeros_like<float>(b);
    nc::NdArray<float> X = nc::zeros_like<float>(b);
    int rows = lower.shape().rows;
    /* L * Y = B --> Y
     * Example: y3 = (b3 - L31 * y1 - L32 * y2) / L33
     */
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < i; j++)
            b(i, 0) -= lower(i, j) * Y(j, 0);
        Y(i, 0) = b(i, 0) / lower(i, i);
    }
    /* L.T * X = Y --> X
     * Example: x1 = (y1 - L21 * x2 - L31 * x3) / L11 */
    for (int i = rows - 1; i >= 0; i--)
    {
        for (int j = i + 1; j < rows; j++)
            Y(i, 0) -= lower(j, i) * X(j, 0);
        X(i, 0) = Y(i, 0) / lower(i, i);
    }
    return X;
}

void KalmanFilter::update(cv::Point measurement)
{
    nc::NdArray<float> projected_mean; // Hk * xk|k-1 shape: (2,1)
    nc::NdArray<float> projected_cov;  // Sk shape: (2,2)
    project(projected_mean, projected_cov);

    /* Cholesky 分解求Kg */
    nc::NdArray<double> lower = nc::linalg::cholesky(projected_cov.transpose());
    // solve Sk.T * Kg.T = (Pk|k-1 * Hk.T).T
    nc::NdArray<float> B = nc::dot(
                               covariance,
                               _update_mat.transpose())
                               .transpose();
    nc::NdArray<float> kalman_gain = nc::zeros_like<float>(B);
    for (int i = 0; i < B.shape().cols; i++)
        kalman_gain.put(kalman_gain.rSlice(), i, cholesky_solve(lower, B(B.rSlice(), i)));
    kalman_gain = kalman_gain.transpose(); // shape:(4,2)

    nc::NdArray<float> innovation = \
        (nc::NdArray<float>){(float)(measurement.x), (float)(measurement.y)}.transpose() - projected_mean; // yk shape:(2,1)
    // xk|k = xk|k-1 + Kg * yk
    mean = mean + nc::dot(kalman_gain, innovation);
    // Pk|k = Pk|k-1 - Kg * Sk * Kg.T 相比于传统公式，计算量小一些
    covariance -= nc::linalg::multi_dot(
        {kalman_gain, projected_cov, kalman_gain.transpose()});
}

void KalmanFilter::get_state(cv::Point &inxy)
{
    nc::NdArray<float> xy = mean(nc::Slice(0, 2), 0);
    inxy.x = xy(0, 0);
    inxy.y = xy(1, 0);

}
