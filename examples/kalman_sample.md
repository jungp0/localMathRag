# Kalman Filter Notes

The discrete linear system is:

$$x_k = F_k x_{k-1} + B_k u_k + w_k$$

where process noise covariance is Q_k.

## Prediction

The covariance prediction equation is:

$$P_{k|k-1}=F_k P_{k-1|k-1} F_k^T + Q_k$$

Here P_{k|k-1} is the predicted covariance, F_k is the state transition matrix,
and Q_k is the process noise covariance.

## Update

Kalman gain:

$$K_k = P_{k|k-1} H_k^T (H_k P_{k|k-1} H_k^T + R_k)^{-1}$$

![Figure 1. Kalman covariance propagation diagram](figures/kalman_covariance.png)

Figure 1 explains how covariance prediction propagates uncertainty through
the state transition matrix before adding process noise.

The posterior covariance should be updated using:

$$P_{k|k}=(I-K_kH_k)P_{k|k-1}$$

The filter shall use R_k as the measurement noise covariance.

max_iterations: 25
sampling_period_ms: 10 ms
