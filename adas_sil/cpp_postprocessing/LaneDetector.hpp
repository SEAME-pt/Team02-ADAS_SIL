#pragma once

#include "opencv2/opencv.hpp"
#include <omp.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <deque>
#include <cmath>

#include "IPM.hpp"


#define WIDTH 384
#define HEIGHT 192

// #define WIDTH 256
// #define HEIGHT 128

struct LaneResult {
  std::vector<double> left_coeffs;  // Polynomial coefficients (a, b, c) for left lane
  std::vector<double> right_coeffs; // Polynomial coefficients for right lane
  std::vector<cv::Point> left_points;  // Points forming the left lane curve
  std::vector<cv::Point> right_points; // Points forming the right lane curve
  std::vector<cv::Point> mid_points;   // Points forming the middle lane
  float lateral_error;              // Normalized lateral error (-1 to 1)
};

class LaneDetector
{
  private:

    void* inputDevice;
    void* outputDevice;
    float* inputData;
    float* outputData;
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    cv::Mat map1, map2;

    std::vector<cv::Point> prevLeftPoints;
    std::vector<cv::Point> prevRightPoints;
    
    cv::Mat leftCoeffs;
    cv::Mat rightCoeffs;


    const int FRAME_SKIP;
    cv::KalmanFilter leftLaneKF, rightLaneKF;
    bool kfInitialized = false;
    double laneWidthEstimate = 0.0;
    bool firstFrame;
    int frame_count;

    std::deque<std::vector<cv::Point>> leftLaneHistory;
    std::deque<std::vector<cv::Point>> rightLaneHistory;
    std::vector<cv::Point> prevLeftCurve;
    std::vector<cv::Point> prevRightCurve;
    std::vector<cv::Point> prevMidCurve;
    cv::Point prevMidPoint = cv::Point(-1, -1);
    const size_t historySize = 5;

    std::vector<cv::Point> allLanePoints;
    cv::Mat lanePointsVisualization;


    // IPM ipm;                // inverse-perspective mapper
    // cv::Size bevSize;       // BEV resolution

  public:
    LaneDetector();
    ~LaneDetector();

    void detect(cv::Mat& frame);
    void run();
    cv::Mat preProcess(const cv::Mat& frame);
    void postProcess(cv::Mat& frame);

    float* getInputData() { return inputData; }
    float* getOutputData() { return outputData; }
    void setOutputData(float* data, size_t size) { 
      std::memcpy(outputData, data, size); 
    }
    
    const cv::Mat& getLeftCoeffs() const { return leftCoeffs; }
    const cv::Mat& getRightCoeffs() const { return rightCoeffs; }
    const std::vector<cv::Point>& getLeftPoints() const { return prevLeftPoints; }
    const std::vector<cv::Point>& getRightPoints() const { return prevRightPoints; }
    const std::vector<cv::Point>& getAllLanePoints() const { return allLanePoints; }
    cv::Mat getLanePointsVisualization() const { return lanePointsVisualization; }
    
    static const int getWIDTH() { return WIDTH; }
    static const int getHEIGHT() { return HEIGHT; }

    void visualizeBothViews(cv::Mat& display_frame);

  private:

    cv::Mat regionOfInterest(const cv::Mat& img,
    const std::vector<cv::Point>& vertices);
    cv::Mat polyfit(const cv::Mat& y_vals, const cv::Mat& x_vals, int degree);
    void createLanes(std::vector<cv::Point> lanes, cv::Mat& frame);

    void drawLanes(cv::Mat& frame, 
      const std::vector<cv::Point>& leftCurve, 
      const std::vector<cv::Point>& rightCurve);

    int cluster2DPoints(const std::vector<cv::Point>& points, 
        std::vector<std::vector<cv::Point>>& clusters,
        float distanceThreshold);

    void clusterLanePoints(const std::vector<cv::Point>& points, 
      std::vector<cv::Point>& leftPoints,
      std::vector<cv::Point>& rightPoints,
      cv::Mat& frame);

    std::vector<cv::Point> fitCurveToPoints(const std::vector<cv::Point>& points, cv::Mat& frame);
    void initKalmanFilters(const std::vector<cv::Point>& leftCurve, 
      const std::vector<cv::Point>& rightCurve);

    void sendCoefs(const std::vector<cv::Point>& leftCurve,
        const std::vector<cv::Point>& rightCurve);
};  