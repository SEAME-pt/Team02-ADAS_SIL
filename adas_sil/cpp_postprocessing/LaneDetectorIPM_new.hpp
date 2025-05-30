#pragma once

#include "opencv2/opencv.hpp"
#include <omp.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <deque>
#include <cmath>

#include <chrono>
#include <unordered_map>

#include "IPM.hpp"

#include "KalmanFilter.hpp"


// #define WIDTH 384
// #define HEIGHT 192

#define WIDTH 256
#define HEIGHT 128

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
    
    IPM *ipm;
    KalmanFilter *kalmanFilter;
    cv::Size bevSize;
    cv::Mat bev_image;

    std::vector<cv::Point> prevLeftCurve;
    std::vector<cv::Point> prevRightCurve;

    cv::Mat allPolylinesViz_;
    int frameWidth_;
    int frameHeight_;
    
    int leftLaneLastUpdatedFrame = 0;
    int rightLaneLastUpdatedFrame = 0;
    int currentFrame = 0;
    const int MAX_LANE_MEMORY_FRAMES = 25;

    cv::Mat leftCoeffs;
    cv::Mat rightCoeffs;
    cv::Mat midCoeffs;

   
    double laneWidthEstimate = 0.0;
    bool firstFrame;
    int frame_count;



    std::vector<cv::Point> allLanePoints;
    cv::Mat lanePointsVisualization;

    float laneError = 0.0f;


    bool ipm_initialized = false;
    cv::Mat original_frame;
    int frameWidth = WIDTH;  // Ensure these are initialized
    int frameHeight = HEIGHT;




  public:
    LaneDetector();
    ~LaneDetector();

 
    cv::Mat preProcess(const cv::Mat& frame);
    void postProcess(cv::Mat& frame);

    float* getInputData() { return inputData; }
    float* getOutputData() { return outputData; }
    void setOutputData(float* data, size_t size) { 
      std::memcpy(outputData, data, size); 
    }
    
    const std::vector<cv::Point>& getAllLanePoints() const { return allLanePoints; }
    cv::Mat getLanePointsVisualization() const { return lanePointsVisualization; }
    
    void visualizeBothViews(cv::Mat& display_frame);
    
    static const int getWIDTH() { return WIDTH; }
    static const int getHEIGHT() { return HEIGHT; }
    
    
    
    const cv::Mat& getLeftCoeffs() const { return leftCoeffs; }
    const cv::Mat& getRightCoeffs() const { return rightCoeffs; }
    cv::Mat getMidCoeffs() const { return midCoeffs; }
    
    const float getLaneError() const { return laneError; }
    cv::Mat getBevImage() const { return bev_image; }
    cv::Mat getPolyLines() const { return allPolylinesViz_; }

  private:

    cv::Mat polyfit(const cv::Mat& y_vals, const cv::Mat& x_vals, int degree);

    std::vector<std::vector<cv::Point>> clusterLaneMask(const cv::Mat& laneMask, int kernelSize, int minArea, int maxLanes);
    void createLanesIPM(std::vector<cv::Point> lanePoints, cv::Mat& frame);
    void mergeLaneComponents(std::vector<std::vector<cv::Point>>& lanePolylines, 
      float maxHorizontalDist, float minOverlapRatio);
    
    void drawPolyLanes(std::vector<std::vector<cv::Point>> lanePolylines);

    float calculateLaneDistance(const std::vector<cv::Point>& lane1, const std::vector<cv::Point>& lane2);
    bool validateLaneSeparation(const std::vector<std::vector<cv::Point>>& lanePolylines, float minLaneWidth);
    void checkPredicedCurve(std::vector<cv::Point>& predictedCurve, const std::vector<cv::Point>& realLane, bool isLeftLane);
    void defineTrajectoryCurve(std::vector<cv::Point>& midCurve, std::vector<cv::Point>& leftCurve, std::vector<cv::Point>& rightCurve);
    void createMidPointError(std::vector<cv::Point>& midCurve, cv::Mat frame);


    void drawLanes(cv::Mat& frame, 
      const std::vector<cv::Point>& leftCurve, 
      const std::vector<cv::Point>& rightCurve);


    void clusterLanePoints(const std::vector<cv::Point>& points, 
      std::vector<cv::Point>& leftPoints,
      std::vector<cv::Point>& rightPoints,
      cv::Mat& frame);

    bool checkIfLeftLane(const std::vector<std::vector<cv::Point>> &lanePolylines);

};