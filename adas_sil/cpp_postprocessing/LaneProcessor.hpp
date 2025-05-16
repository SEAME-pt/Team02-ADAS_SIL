#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <queue>
#include <memory>

struct LaneResult {
    std::vector<double> left_coeffs;  // Polynomial coefficients (a, b, c) for left lane
    std::vector<double> right_coeffs; // Polynomial coefficients for right lane
    std::vector<cv::Point> left_points;  // Points forming the left lane curve
    std::vector<cv::Point> right_points; // Points forming the right lane curve
    std::vector<cv::Point> mid_points;   // Points forming the middle lane
    float lateral_error;              // Normalized lateral error (-1 to 1)
};

class LaneProcessor {
public:
    LaneProcessor(int width, int height);
    
    // Process a binary mask and extract lane curves
    LaneResult process(const cv::Mat& mask);
    
    // Draw detected lanes onto a visualization frame
    void drawLanes(cv::Mat& frame, const LaneResult& result);

private:
    int width;
    int height;
    bool firstFrame;
    float laneWidthEstimate;
    float lateralError;

    
    // Previous state for temporal filtering
    std::vector<cv::Point> prevLeftCurve;
    std::vector<cv::Point> prevRightCurve;
    std::vector<cv::Point> prevLeftPoints;
    std::vector<cv::Point> prevRightPoints;
    std::vector<cv::Point> midCurve;
    cv::Point prevMidPoint;
    cv::Point midPoint;
    
    // Core processing functions
    void createLanes(std::vector<cv::Point>& lanePoints, LaneResult& result);
    void clusterLanePoints(const std::vector<cv::Point>& points,
                         std::vector<cv::Point>& leftPoints,
                         std::vector<cv::Point>& rightPoints);
    std::vector<cv::Point> fitCurveToPoints(const std::vector<cv::Point>& points);
    cv::Mat polyfit(const cv::Mat& y_vals, const cv::Mat& x_vals, int degree);
    int cluster2DPoints(const std::vector<cv::Point>& points, 
                      std::vector<std::vector<cv::Point>>& clusters,
                      float distanceThreshold);
};