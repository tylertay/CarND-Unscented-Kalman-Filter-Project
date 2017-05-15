#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Initial state vector
  x_ = VectorXd(n_x_);

  // Initial state covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // Initial predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.6; // tune this parameter using NIS

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.6; // tune this parameter using NIS

  // Statistic table
  // std_a_, std_yawdd_, laser NIS statistic, radar NIS statistic, rmse (4 values)
  // 0.5, 0.5, 0.012, 0.06, 0.061, 0.086, 0.330, 0.213
  // 0.6, 0.5, 0.012, 0.056, 0.062, 0.085, 0.330, 0.213
  // 0.6, 0.6, 0.012, 0.056, 0.061, 0.085, 0.0330, 0.212
  // 0.6, 0.8, 0.012, 0.052, 0.061, 0.085, 0.331, 0.214
  // 3, 0.5, 0.016, 0.056, 0.0740, 0.085, 0.356, 0.241

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // Weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Initialize state vector
  x_ << 0, 0, 0, 0, 0;

  // Initialize state covariance matrix
  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  // Initialization
  if (!is_initialized_) {
    // Initialize augmented state covariance matrix

    // Initialize weights
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i = 1; i < 2 * n_aug_ + 1; i++) {
      weights_(i) = 0.5 / (n_aug_ + lambda_);
    }

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      double rho = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double rhod = meas_package.raw_measurements_(2);

      double vx = rhod * cos(phi);
      double vy = rhod * sin(phi);

      x_(0) = rho * cos(phi);
      x_(1) = rho * sin(phi);
      x_(2) = sqrt(vx * vx + vy * vy);
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_(0) = meas_package.raw_measurements_(0);
      x_(1) = meas_package.raw_measurements_(1);
    }

    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;
    return;
  }

  // Prediction

  // Time elapsed (in seconds)
  float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  Prediction(dt);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  // 1. Generate sigma points

  // Sigma points
  MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);
  // Square root of P
  MatrixXd sr_P = P_.llt().matrixL();
  // Set first sigma point
  Xsig.col(0) = x_;
  // Set remaining sigma points
  double a = sqrt(lambda_ + n_x_);
  for (int i = 0; i < n_x_; i++) {
    MatrixXd o = a * sr_P.col(i);
    Xsig.col(i + 1) = x_ + o;
    Xsig.col(i + 1 + n_x_) = x_ - o;
  }


  // 2. Generate augmented state and augmented state covariance

  // Spreading parameter for augmented
  int lambda = 3 - n_aug_;

  // Augmented mean vector
  VectorXd x_aug = VectorXd(7);

  // Augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  // Sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // Augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // Augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  // Square root of augmented P
  MatrixXd sr_P_aug = P_aug.llt().matrixL();

  // Create augmented
  Xsig_aug.col(0) = x_aug;
  a = sqrt(lambda + n_aug_);
  for (int i = 0; i < n_aug_; i++) {
    MatrixXd o = a * sr_P_aug.col(i);
    Xsig_aug.col(i + 1) = x_aug + o;
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - o;
  }


  // 3. Predict sigma points

  // Predict sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double px = Xsig_aug(0,i);
    double py = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    // Avoid division by zero
    if (fabs(yawd) > .001) {
      px += v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py += v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    }
    else {
      px += v * delta_t * cos(yaw);
      py += v * delta_t * sin(yaw);
    }

    // Add noise
    px += 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py += 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v += delta_t * nu_a;
    yaw += yawd * delta_t + 0.5 * (delta_t * delta_t) * nu_yawdd;
    yawd += delta_t * nu_yawdd;

    Xsig_pred_(0,i) = px;
    Xsig_pred_(1,i) = py;
    Xsig_pred_(2,i) = v;
    Xsig_pred_(3,i) = yaw;
    Xsig_pred_(4,i) = yawd;
  }


  // 4. Predict state mean and state covariance

  // Predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  // Predicted state covariance
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // Angle normalization
    if (x_diff(3) > M_PI) {
      x_diff(3) -= 2 * M_PI;
    }
    if (x_diff(3) < -M_PI) {
      x_diff(3) += 2 * M_PI;
    }

    P_ += weights_(i) * x_diff * x_diff.transpose();
  }

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * Updates lidar NIS
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  // Measurements
  VectorXd z = meas_package.raw_measurements_;

  // Measurement dimension (px, py)
  int n_z = 2;

  // Sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // Measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // Sigma point predictions in process space
    double px = Xsig_pred_(0,i);
    double py = Xsig_pred_(1,i);

    // Sigma point predictions in measurement space
    Zsig(0,i) = px;
    Zsig(1,i) = py;
  }

  // Calculate mean predicted measurement
  z_pred.fill(0);
  z_pred = Zsig * weights_;

  // Calculate measurement covariance
  S.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    if (z_diff(1) > M_PI)
      z_diff(1) -= 2 * M_PI;
    if (z_diff(1) < -M_PI)
      z_diff(1) += 2 * M_PI;
    S += weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R << std_laspx_ * std_laspx_, 0,
       0, std_laspy_ * std_laspy_;
  S = S + R;

  // Cross correlation
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // Calculate cross correlation
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    if (z_diff(1) > M_PI) {
      z_diff(1) -= 2 * M_PI;
    }
    if (z_diff(1) < -M_PI) {
      z_diff(1) += 2 * M_PI;
    }

    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // Angle normalization
    if (x_diff(3) > M_PI) {
      x_diff(3) -= 2 * M_PI;
    }
    if (x_diff(3) < -M_PI) {
      x_diff(3) += 2 * M_PI;
    }

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain
  MatrixXd K = Tc * S.inverse();

  // Residual
  VectorXd z_diff = z - z_pred;

  if (z_diff(1) > M_PI) {
    z_diff(1) -= 2 * M_PI;
  }
  if (z_diff(1) < -M_PI) {
    z_diff(1) += 2 * M_PI;
  }

  // Update state mean
  x_ += K * z_diff;

  // Update covariance
  P_ -= K * S * K.transpose();

  // Calculate NIS
  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * Updates radar NIS
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // Measurements
  VectorXd z = meas_package.raw_measurements_;

  // Measurement dimension (rho, phi, rhod)
  int n_z = 3;

  // Sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  // Measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  // Transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double px = Xsig_pred_(0,i);
    double py = Xsig_pred_(1,i);
    double v = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);
    double yawd = Xsig_pred_(4,i);

    Zsig(0,i) = sqrt(px * px + py * py);
    Zsig(1,i) = atan2(py,px);
    Zsig(2,i) = (px * cos(yaw) * v + py * sin(yaw) * v) / Zsig(0,i);
  }

  // Calculate mean predicted measurement
  z_pred.fill(0);
  z_pred = Zsig * weights_;

  // Calculate measurement covariance
  S.fill(0);
  for (int i = 0; i < Xsig_pred_.cols(); i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    if (z_diff(1) > M_PI)
      z_diff(1) -= 2 * M_PI;
    if (z_diff(1) < -M_PI)
      z_diff(1) += 2 * M_PI;
    S += weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R << std_radr_ * std_radr_, 0, 0,
       0, std_radphi_ * std_radphi_, 0,
       0, 0, std_radrd_ * std_radrd_;
  S = S + R;


  // Cross correlation
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // Calculate cross correlation
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    if (z_diff(1) > M_PI) {
      z_diff(1) -= 2 * M_PI;
    }
    if (z_diff(1) < -M_PI) {
      z_diff(1) += 2 * M_PI;
    }

    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // Angle normalization
    if (x_diff(3) > M_PI) {
      x_diff(3) -= 2 * M_PI;
    }
    if (x_diff(3) < -M_PI) {
      x_diff(3) += 2 * M_PI;
    }

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain
  MatrixXd K = Tc * S.inverse();

  // Residual
  VectorXd z_diff = z - z_pred;

  if (z_diff(1) > M_PI) {
    z_diff(1) -= 2 * M_PI;
  }
  if (z_diff(1) < -M_PI) {
    z_diff(1) += 2 * M_PI;
  }

  // Update state mean
  x_ += K * z_diff;

  // Update covariance
  P_ -= K * S * K.transpose();

  // Calculate NIS
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}
