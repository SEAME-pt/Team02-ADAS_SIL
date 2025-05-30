#pragma once

#include "opencv2/opencv.hpp"

class KalmanFilter
{
  private:
    cv::KalmanFilter rightLaneKF;
    cv::KalmanFilter leftLaneKF;
    cv::KalmanFilter middleLaneKF;

  public:
    cv::Mat middleLaneCoeffs;
    cv::Mat leftLaneCoeffs;
    cv::Mat rightLaneCoeffs;

    bool leftLaneInitialized;
    bool rightLaneInitialized;
    bool midLaneInitialized;

  public:
    KalmanFilter();
    ~KalmanFilter();

    void init();

    void updateLeftLaneFilter(const std::vector<cv::Point>& lane);
    void updateRightLaneFilter(const std::vector<cv::Point>& lane);
    void updateMiddleLaneFilter(const std::vector<cv::Point>& lane);
    std::vector<cv::Point> predictLeftLaneCurve(int height, int width);
    std::vector<cv::Point> predictRightLaneCurve(int height, int width);
    std::vector<cv::Point> predictMiddleLaneCurve(int height, int width);

  private:
  
    std::vector<cv::Point> reconstructLaneFromCoefficients(const cv::Mat& coeffs, int height, int width);
    cv::Mat extractPolynomialCoefficients(const std::vector<cv::Point>& laneCurve);
    cv::Mat polyfit(const cv::Mat& y_vals, const cv::Mat& x_vals, int degree);

};