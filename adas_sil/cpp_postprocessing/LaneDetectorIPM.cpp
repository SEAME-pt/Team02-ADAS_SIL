#include "LaneDetectorIPM.hpp"

#include <iostream>

using namespace cv;
using namespace std;

LaneDetector::LaneDetector()
    : FRAME_SKIP(3),
      laneWidthEstimate(0.0), firstFrame(true), frame_count(0)
{
    // set kalman filter status to false
    kfInitialized = false;
    inputData = new float[3 * HEIGHT * WIDTH];
    outputData = new float[1 * HEIGHT * WIDTH];

    std::memset(inputData, 0, 3 * HEIGHT * WIDTH * sizeof(float));
    std::memset(outputData, 0, 1 * HEIGHT * WIDTH * sizeof(float));
    std::cout << "LaneDetector initialized with inputData and outputData buffers." << std::endl;
}

LaneDetector::~LaneDetector()
{
    delete[] inputData;
    delete[] outputData;
}

// Polynomial fitting using OpenCV
Mat LaneDetector::polyfit(const Mat& y_vals, const Mat& x_vals, int degree)
{
    // Check if we have enough points for the requested degree
    if (y_vals.rows < degree + 1)
    {
        // Not enough points - reduce degree or return empty matrix
        if (y_vals.rows < 2)
        {
            return Mat();
        }
        // Adjust degree based on available points
        degree = y_vals.rows - 1;
    }

    // Ensure data is in the right format (CV_64F)
    Mat y_vals_64f, x_vals_64f;
    y_vals.convertTo(y_vals_64f, CV_64F);
    x_vals.convertTo(x_vals_64f, CV_64F);

    // Create the design matrix with appropriate dimensions
    Mat A = Mat::zeros(y_vals_64f.rows, degree + 1, CV_64F);

    // Fill the design matrix
    for (int i = 0; i < y_vals_64f.rows; i++)
    {
        for (int j = 0; j <= degree; j++)
        {
            A.at<double>(i, j) = pow(y_vals_64f.at<double>(i), degree - j);
        }
    }

    // Check for invalid values (NaN, Inf)
    for (int i = 0; i < A.rows; i++)
    {
        for (int j = 0; j < A.cols; j++)
        {
            if (cvIsNaN(A.at<double>(i, j)) || cvIsInf(A.at<double>(i, j)))
            {
                A.at<double>(i, j) = 0;
            }
        }
    }

    // Solve the system using SVD for better stability
    Mat coeffs;
    try
    {
        solve(A, x_vals_64f, coeffs, DECOMP_SVD);
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "Error in polyfit: " << e.what() << std::endl;
        return Mat();
    }
    if (abs(coeffs.at<double>(0)) < 0.0005)
        coeffs.at<double>(0) = 0;
    if (abs(coeffs.at<double>(1)) < 0.001)
        coeffs.at<double>(1) = 0;
    if (abs(coeffs.at<double>(2)) < 0.001)
        coeffs.at<double>(2) = 0;

    return coeffs;
}


cv::Mat LaneDetector::preProcess(const cv::Mat& frame)
{
    static cv::Mat resized(HEIGHT, WIDTH, CV_8UC3);
    static cv::Mat float_mat(HEIGHT, WIDTH, CV_32FC3);
    
    // Use INTER_NEAREST for faster resizing
    cv::resize(frame, resized, cv::Size(WIDTH, HEIGHT), 0, 0, cv::INTER_NEAREST);
    cv::Mat rgb_image;
    cv::cvtColor(resized, rgb_image, cv::COLOR_BGR2RGB);
    
    return rgb_image;
}

void LaneDetector::postProcess(cv::Mat& frame)
{
    // Apply IPM to get bird's-eye view
    if (!ipm_initialized) {
        // Initialize IPM with default values or camera parameters
        // These values would need to be calibrated for your specific camera
        float cameraHeight = 1.5f;       // meters
        float cameraPitch = 15.0f;       // degrees down from horizontal
        float horizontalFOV = 105.0f;     // degrees
        float img_height = static_cast<float>(HEIGHT);
        float img_width = static_cast<float>(WIDTH);
        float h_fov_rad = horizontalFOV * CV_PI / 180.0f;
        float verticalFOV = 2.0f * std::atan((img_height/img_width) * std::tan(h_fov_rad/2.0f)) * 180.0f / CV_PI;
        float nearDistance = 1.5f;       // meters
        float farDistance = 15.0f;       // meters
        float laneWidth = 7.0f;          // meters
        bevSize = cv::Size(WIDTH, WIDTH);
        cv::Size origSize = cv::Size(WIDTH, HEIGHT);
        ipm.initialize(origSize, bevSize);
        ipm.calibrateFromCamera(cameraHeight, cameraPitch, horizontalFOV, verticalFOV,
                                nearDistance, farDistance, laneWidth);
        ipm_initialized = true;
    }
    
    // Store the original image for visualization and debugging
    original_frame = frame.clone();
    
    // Apply IPM to get bird's-eye view
    // cv::Mat bev_image = ipm.applyIPM(frame);
    
    static cv::Mat mask(HEIGHT, WIDTH, CV_8UC1);

    uchar* mask_data       = mask.data;
    const int total_pixels = HEIGHT * WIDTH;

    for (int i = 0; i < total_pixels; i++)
    {
        mask_data[i] = (outputData[i] > 0.5) ? 255 : 0;
    }

    // cv::Mat bev_mask = ipm.applyIPM(mask);
    cv::Mat bev_mask = ipm.transformPoints(mask);
    // Create a copy of the mask for ROI application
    cv::Mat roiMask = bev_mask.clone();
    std::vector<cv::Point> points;
   
    // Apply ROI directly to the mask before collecting points
    int height = roiMask.rows;
    int width = roiMask.cols;

        // Define trapezoidal ROI (same as in filterPoints)
    float topY = height * 0.1f; 
    float bottomY = height * 0.95f;
    
    float topLeftX = width * 0.2f;
    float topRightX = width * 0.8f;
    float bottomLeftX = width * 0.05f;
    float bottomRightX = width * 0.95f;

    // Set all points outside ROI to zero
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Calculate if this point is inside the trapezoid
            bool inROI = false;
            if (y >= topY) {
                float yRatio = (y - topY) / (bottomY - topY);
                float leftBoundary = topLeftX + yRatio * (bottomLeftX - topLeftX);
                float rightBoundary = topRightX + yRatio * (bottomRightX - topRightX);
                
                inROI = (x >= leftBoundary && x <= rightBoundary);
            }
            
            // If outside ROI, set to zero
            if (!inROI) {
                roiMask.at<uchar>(y, x) = 0;
            }
        }
    }

    // Collect points from the ROI-filtered mask
    std::vector<cv::Point> maskPoints;
    cv::findNonZero(roiMask, maskPoints);

    // Scale all points to frame coordinates
    std::vector<cv::Point> lanePoints;
    
    if (mask.cols == frame.cols && mask.rows == frame.rows) {
        // No scaling needed - use points directly
        lanePoints = maskPoints;
    } else {
        // Scale points to frame coordinates if sizes differ
        lanePoints.reserve(maskPoints.size());
        float x_scale = static_cast<float>(frame.cols) / mask.cols;
        float y_scale = static_cast<float>(frame.rows) / mask.rows;
        
        for (const auto& pt : maskPoints) {
            int scaledX = pt.x * x_scale;
            int scaledY = pt.y * y_scale;
            lanePoints.push_back(cv::Point(scaledX, scaledY));
        }
    }
    // std::cout << "lanepoints number = " << lanePoints.size() << std::endl;
    
    // Draw the ROI for visualization (optional)
    // std::vector<cv::Point> trapezoid = {
    //     cv::Point(bottomLeftX, bottomY),
    //     cv::Point(bottomRightX, bottomY),
    //     cv::Point(topRightX, topY),
    //     cv::Point(topLeftX, topY)
    // };
    // cv::polylines(bev_image, std::vector<std::vector<cv::Point>>{trapezoid}, 
    //              true, cv::Scalar(0, 255, 255), 2);
    
    // Create a proper BEV image
    cv::Mat bev_image = cv::Mat(bevSize, CV_8UC3, cv::Scalar(0, 0, 0));

    // Create mask and process points as before...
    
    // Draw detected points on BEV image
    for (const auto& pt : lanePoints) {
        cv::circle(bev_image, pt, 3, cv::Scalar(255, 255, 255), -1);
    }

    this->bev_image = bev_image;

    createLanesIPM(lanePoints, bev_image);
    // this->bev_image = bev_image;
    
}

std::vector<std::vector<cv::Point>> LaneDetector::processLaneMask(const cv::Mat& laneMask, int kernelSize, int minArea, int maxLanes) {
    // Ensure binary mask
    cv::Mat binaryMask;
    if (laneMask.channels() == 3) {
        cv::cvtColor(laneMask, binaryMask, cv::COLOR_BGR2GRAY);
    } else {
        binaryMask = laneMask.clone();
    }
    
    static cv::Mat verticalKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize * 3));
    
    // Create proper diagonal kernels
    static cv::Mat diagonalKernel1; // Right curve (sloped right)
    static cv::Mat diagonalKernel2; // Left curve (sloped left)
    
    // Initialize diagonal kernels if not already created
    if (diagonalKernel1.empty() || diagonalKernel2.empty()) {
        // Create diagonal kernel for right curves (top-left to bottom-right)
        diagonalKernel1 = cv::Mat::zeros(kernelSize * 3, kernelSize * 3, CV_8U);
        for (int i = 0; i < kernelSize * 3; i++) {
            for (int j = 0; j < kernelSize; j++) {
                // Create a curved pattern
                int offset = std::pow(i / (kernelSize * 3.0) - 0.5, 2) * kernelSize * 8;
                int x = i/2 + j + offset;
                if (x >= 0 && x < kernelSize * 3)
                    diagonalKernel1.at<uchar>(i, x) = 1;
            }
        }
        
        // Create diagonal kernel for left curves (top-right to bottom-left) 
        diagonalKernel2 = cv::Mat::zeros(kernelSize * 3, kernelSize * 3, CV_8U);
        for (int i = 0; i < kernelSize * 3; i++) {
            for (int j = 0; j < kernelSize; j++) {
                // Create a curved pattern (mirror of the first one)
                int offset = std::pow(i / (kernelSize * 3.0) - 0.5, 2) * kernelSize * 8;
                int x = kernelSize * 3 - 1 - (i/2 + j + offset);
                if (x >= 0 && x < kernelSize * 3)
                    diagonalKernel2.at<uchar>(i, x) = 1;
            }
        }
    }
    static cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    
    // Apply multiple directional closings to better connect curved dashed lines
    cv::Mat result = binaryMask.clone();
    cv::morphologyEx(result, result, cv::MORPH_CLOSE, verticalKernel);
    
    // IMPORTANT: Only apply diagonal kernels if we have curve lanes (check prevLeftCurve/prevRightCurve)
    if (!prevLeftCurve.empty() && prevLeftCurve.size() >= 3) {
        // Estimate curvature from previous frame
        double curvature = estimateCurvature(prevLeftCurve);
        if (std::abs(curvature) > 0.0001) {  // If significant curvature
            // Choose different kernels based on curve direction
            if (curvature > 0) {
                // Right curve - use diagonal kernel
                cv::morphologyEx(result, result, cv::MORPH_CLOSE, diagonalKernel1);
            } else {
                // Left curve - use other diagonal kernel
                cv::morphologyEx(result, result, cv::MORPH_CLOSE, diagonalKernel2);
            }
        }
    }
    
    // Standard kernel closing (efficient reuse)
    cv::morphologyEx(result, result, cv::MORPH_CLOSE, kernel);

    // Rest of the function remains mostly the same
    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(result, labels, stats, centroids, 8, CV_32S);
    
    // OPTIMIZATION: Pre-allocate with approximate capacity
    std::vector<std::pair<int, float>> validComponents;
    validComponents.reserve(std::min(numLabels, maxLanes + 3));
    
    for (int i = 1; i < numLabels; i++) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > minArea) {
            float centerX = centroids.at<double>(i, 0);
            validComponents.push_back(std::make_pair(i, centerX));
        }
    }
    
    // OPTIMIZATION: Use partial sort instead of full sort when possible
    if (validComponents.size() > maxLanes) {
        std::partial_sort(validComponents.begin(), validComponents.begin() + maxLanes, 
                        validComponents.end(), 
                        [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                            return a.second < b.second;
                        });
        validComponents.resize(maxLanes);
    } else {
        std::sort(validComponents.begin(), validComponents.end(), 
            [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                return a.second < b.second;
            });
    }
    
    // OPTIMIZATION: Reserve capacity for output
    std::vector<std::vector<cv::Point>> lanePolylines;
    lanePolylines.reserve(validComponents.size());
    
    // Process each lane with optimized extraction
    for (const auto& comp : validComponents) {
        int compIdx = comp.first;
        
        // OPTIMIZATION: Extract points more efficiently using row pointers
        std::vector<cv::Point> lanePoints;
        lanePoints.reserve(labels.rows/5); // Pre-allocate approx size
        
        for (int y = 0; y < labels.rows; y += 2) {
            const int* row = labels.ptr<int>(y);
            int xStart = -1, xEnd = -1;
            
            for (int x = 0; x < labels.cols; x++) {
                if (row[x] == compIdx) {
                    if (xStart < 0) xStart = x;
                    xEnd = x;
                }
            }
            
            if (xStart >= 0) {
                int midX = (xStart + xEnd) / 2;
                lanePoints.push_back(cv::Point(midX, y));
            }
        }
        
        if (!lanePoints.empty()) {
            lanePolylines.push_back(std::move(lanePoints)); // Use move semantics
        }
    }
    
    return lanePolylines;
}

// Helper function to estimate curvature from a set of points
double LaneDetector::estimateCurvature(const std::vector<cv::Point>& points) {
    if (points.size() < 3) return 0.0;
    
    std::vector<float> x_vals, y_vals;
    for (const auto& pt : points) {
        x_vals.push_back(pt.x);
        y_vals.push_back(pt.y);
    }
    
    cv::Mat x_mat(x_vals), y_mat(y_vals);
    cv::Mat coeffs = polyfit(y_mat, x_mat, 2);
    
    if (!coeffs.empty() && coeffs.rows >= 3) {
        return coeffs.at<double>(0);  // Return the quadratic coefficient (curvature)
    }
    
    return 0.0;
}


void LaneDetector::clusterLanePointsOnCurve(const std::vector<cv::Point>& points, 
                                         std::vector<cv::Point>& leftPoints,
                                         std::vector<cv::Point>& rightPoints) {
    leftPoints.clear();
    rightPoints.clear();
    
    if (points.empty()) return;
    
    // When we have previous lanes, use them as seeds for clustering
    if (!prevLeftCurve.empty() && !prevRightCurve.empty()) {
        // Pre-compute left and right lane polynomials
        std::vector<float> left_x, left_y, right_x, right_y;
        for (const auto& pt : prevLeftCurve) {
            left_x.push_back(pt.x);
            left_y.push_back(pt.y);
        }
        
        for (const auto& pt : prevRightCurve) {
            right_x.push_back(pt.x);
            right_y.push_back(pt.y);
        }
        
        cv::Mat leftX(left_x), leftY(left_y), rightX(right_x), rightY(right_y);
        cv::Mat leftCoeffs = polyfit(leftY, leftX, 2);
        cv::Mat rightCoeffs = polyfit(rightY, rightX, 2);
        
        // If polynomials are valid
        if (!leftCoeffs.empty() && !rightCoeffs.empty() && 
            leftCoeffs.rows >= 3 && rightCoeffs.rows >= 3) {
            
            // Calculate adaptive threshold based on lane width
            double laneWidth = laneWidthEstimate;
            double threshold = laneWidth * 0.4; // 40% of lane width
            
            // Assign points to left or right based on distance to curves
            for (const auto& pt : points) {
                // Compute expected x positions on both curves at this y
                double leftX = leftCoeffs.at<double>(0) * pt.y * pt.y + 
                             leftCoeffs.at<double>(1) * pt.y + 
                             leftCoeffs.at<double>(2);
                             
                double rightX = rightCoeffs.at<double>(0) * pt.y * pt.y + 
                              rightCoeffs.at<double>(1) * pt.y + 
                              rightCoeffs.at<double>(2);
                
                // Calculate distances to both curves
                double leftDist = std::abs(pt.x - leftX);
                double rightDist = std::abs(pt.x - rightX);
                
                // Assign to closest curve if within threshold
                if (leftDist < rightDist && leftDist < threshold) {
                    leftPoints.push_back(pt);
                } else if (rightDist < leftDist && rightDist < threshold) {
                    rightPoints.push_back(pt);
                }
                // Points not close to either curve are ignored
            }
            
            return;
        }
    }
    
    // Fallback: use simple x-based clustering
    int width = 0;
    for (const auto& pt : points) {
        width = std::max(width, pt.x);
    }
    
    int midX = width / 2;
    for (const auto& pt : points) {
        if (pt.x < midX) {
            leftPoints.push_back(pt);
        } else {
            rightPoints.push_back(pt);
        }
    }
}


// Add this function after processLaneMask
void LaneDetector::mergeLaneComponents(std::vector<std::vector<cv::Point>>& lanePolylines, float maxHorizontalDist, float minOverlapRatio) {
    if (lanePolylines.size() <= 1) return;
    
    bool mergePerformed = true;
    while (mergePerformed) {
        mergePerformed = false;
        
        for (size_t i = 0; i < lanePolylines.size() && !mergePerformed; i++) {
            for (size_t j = i + 1; j < lanePolylines.size() && !mergePerformed; j++) {
                // Compute the y-range of both polylines
                int minY1 = INT_MAX, maxY1 = 0;
                int minY2 = INT_MAX, maxY2 = 0;
                float avgX1 = 0, avgX2 = 0;
                
                for (const auto& pt : lanePolylines[i]) {
                    minY1 = std::min(minY1, pt.y);
                    maxY1 = std::max(maxY1, pt.y);
                    avgX1 += pt.x;
                }
                avgX1 /= lanePolylines[i].size();
                
                for (const auto& pt : lanePolylines[j]) {
                    minY2 = std::min(minY2, pt.y);
                    maxY2 = std::max(maxY2, pt.y);
                    avgX2 += pt.x;
                }
                avgX2 /= lanePolylines[j].size();
                
                // Check if they're horizontally aligned
                float hDist = std::abs(avgX1 - avgX2);
                
                // Check for vertical relationship (one above the other)
                bool verticallyAligned = (minY1 > maxY2) || (minY2 > maxY1);
                
                if (hDist <= maxHorizontalDist && verticallyAligned) {
                    // Merge polylines
                    lanePolylines[i].insert(lanePolylines[i].end(), 
                                            lanePolylines[j].begin(), 
                                            lanePolylines[j].end());
                    lanePolylines.erase(lanePolylines.begin() + j);
                    mergePerformed = true;
                    break;
                }
            }
        }
    }
    
    // Sort points in each polyline by y-coordinate
    for (auto& polyline : lanePolylines) {
        std::sort(polyline.begin(), polyline.end(), 
            [](const cv::Point& a, const cv::Point& b) {
                return a.y < b.y;
            });
    }
}

void LaneDetector::createLanesIPM(std::vector<cv::Point> lanePoints,
    cv::Mat& frame)
{

    std::chrono::steady_clock::time_point start_total = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point section_start, section_end;
    std::unordered_map<std::string, double> timings;

    if (firstFrame) {
        laneWidthEstimate = frame.cols * 0.25;
        firstFrame = false;
    }
    
    // Create a binary mask from lanePoints
    cv::Mat laneMask = cv::Mat::zeros(frame.size(), CV_8UC1);
    for (const auto& pt : lanePoints) {
        if (pt.x >= 0 && pt.x < laneMask.cols && pt.y >= 0 && pt.y < laneMask.rows) {
            laneMask.at<uchar>(pt.y, pt.x) = 255;
        }
    }
    
    // Process the binary mask to get lane polylines
    std::vector<std::vector<cv::Point>> lanePolylines = processLaneMask(laneMask, 20, 30, 10);
    // std::cout << "Number of lane polylines after merging: " << lanePolylines.size() << std::endl;
    allPolylinesViz = frame.clone();
    std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 0, 0),    // Blue
        cv::Scalar(0, 255, 0),    // Green
        cv::Scalar(0, 0, 255),    // Red
        cv::Scalar(255, 255, 0),  // Cyan
        cv::Scalar(255, 0, 255),  // Magenta
        cv::Scalar(0, 255, 255)   // Yellow
    };
    
    // Draw each polyline with a different color
    for (size_t i = 0; i < lanePolylines.size(); i++) {
        cv::Scalar color = colors[i % colors.size()];
        for (size_t j = 1; j < lanePolylines[i].size(); j++) {
            cv::line(allPolylinesViz, lanePolylines[i][j-1], lanePolylines[i][j], color, 2);
        }
        
        // Add a label for each polyline
        if (!lanePolylines[i].empty()) {
            std::string label = "Lane " + std::to_string(i+1);
            cv::putText(allPolylinesViz, label, lanePolylines[i][0], 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
        }
    }
    
    // Display the number of polylines found
    std::string countText = "Polylines: " + std::to_string(lanePolylines.size());
    cv::putText(allPolylinesViz, countText, cv::Point(20, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    section_start = std::chrono::steady_clock::now();
    float maxHorizontalDistance = frame.cols * 0.05; // 5% of frame width
    mergeLaneComponents(lanePolylines, maxHorizontalDistance, 0.0);
    

    // Take only the largest 2 components after merging
    if (lanePolylines.size() > 2) {
        // Sort by number of points (largest first)
        std::sort(lanePolylines.begin(), lanePolylines.end(), 
            [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                return a.size() > b.size();
            });
        lanePolylines.resize(2);
    }

    section_end = std::chrono::steady_clock::now();
    timings["1. Binary mask"] = std::chrono::duration_cast<std::chrono::microseconds>(section_end - section_start).count() / 1000.0;
    

    section_start = std::chrono::steady_clock::now();

    // If we found exactly 2 lanes, use them directly
    std::vector<cv::Point> leftCurve, rightCurve;
    
    if (lanePolylines.size() == 2) {
        // Find the lowest point (highest y-value) in each polyline
        cv::Point lowestPoint1(-1, -1);
        cv::Point lowestPoint2(-1, -1);
        
        // Find lowest point in first polyline
        for (const auto& pt : lanePolylines[0]) {
            if (pt.y > lowestPoint1.y) {
                lowestPoint1 = pt;
            }
        }
        
        // Find lowest point in second polyline
        for (const auto& pt : lanePolylines[1]) {
            if (pt.y > lowestPoint2.y) {
                lowestPoint2 = pt;
            }
        }
        
        // Determine left and right lanes based on the x-coordinate of lowest points
        int centerX = frame.cols / 2;
        
        // Debug visualization of lowest points
        cv::circle(frame, lowestPoint1, 8, cv::Scalar(255, 0, 255), -1);
        cv::circle(frame, lowestPoint2, 8, cv::Scalar(0, 255, 255), -1);
        
        // Compare x-coordinates to determine left/right
        if (lowestPoint1.x < lowestPoint2.x) {
            leftCurve = lanePolylines[0];
            rightCurve = lanePolylines[1];
            
            // Debug text
            std::string leftText = "Left: " + std::to_string(lowestPoint1.x);
            std::string rightText = "Right: " + std::to_string(lowestPoint2.x);
            cv::putText(frame, leftText, lowestPoint1 + cv::Point(10, 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 255), 1);
            cv::putText(frame, rightText, lowestPoint2 + cv::Point(10, 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        } else {
            leftCurve = lanePolylines[1];
            rightCurve = lanePolylines[0];
            
            // Debug text
            std::string leftText = "Left: " + std::to_string(lowestPoint2.x);
            std::string rightText = "Right: " + std::to_string(lowestPoint1.x);
            cv::putText(frame, leftText, lowestPoint2 + cv::Point(10, 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 255), 1);
            cv::putText(frame, rightText, lowestPoint1 + cv::Point(10, 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        }
    }
    else {
        // Fall back to clustering when no lanes or too many lanes detected
        std::vector<cv::Point> leftPoints, rightPoints;
        // std::cout << "-------- Clustering Curve ----------" << std::endl;
        clusterLanePointsOnCurve(lanePoints, leftPoints, rightPoints);
        
        // Fit polynomial curves to each set of points
        if (leftPoints.size() >= 3) {
            leftCurve = fitCurveToPoints(leftPoints, frame);
        }
        if (rightPoints.size() >= 3) {
            rightCurve = fitCurveToPoints(rightPoints, frame);
        }
    }
    
    section_end = std::chrono::steady_clock::now();
    timings["2. Point Clustering"] = std::chrono::duration_cast<std::chrono::microseconds>(section_end - section_start).count() / 1000.0;
    section_start = std::chrono::steady_clock::now();
    // Save lane points for external access
    this->allLanePoints = lanePoints;
    
    // Create visualization frame to show all lane points
    cv::Mat pointsVisualization = frame.clone();
    cv::resize(pointsVisualization, this->lanePointsVisualization, frame.size(), 0, 0, cv::INTER_NEAREST);
    

    // 3. Apply Kalman filtering for temporal smoothing and prediction
    // Initialize Kalman filter if this is the first good detection
    if (!kfInitialized && !leftCurve.empty() && !rightCurve.empty())
    {
        initKalmanFilters(leftCurve, rightCurve);
        std::cout << "Kalman filter initialized" << std::endl;
    }

    // Use Kalman filter predictions when lanes disappear or are unstable
    if (kfInitialized)
    {
        cv::Mat leftPredicted  = leftLaneKF.predict();
        cv::Mat rightPredicted = rightLaneKF.predict();
        
        if (!leftCurve.empty() && leftCurve.size() >= 3)
        {
            // If valid measurement exists, update Kalman filter normally.
            int bottom_idx = 0;
            int mid_idx    = leftCurve.size() / 2;
            int top_idx    = leftCurve.size() - 1;
            cv::Mat measurement = (cv::Mat_<float>(3, 1) << 
                                    leftCurve[bottom_idx].x,
                                    leftCurve[mid_idx].x,
                                    leftCurve[top_idx].x);
            leftLaneKF.correct(measurement);
        }
        else  // Left lane is lost or insufficient
        {
            std::vector<cv::Point> predictedLeftCurve;
            int height = frame.rows;
            
            if (!rightCurve.empty() && rightCurve.size() >= 3)
            {
                // Use the good right lane to predict the missing left lane.
                // Fit a polynomial to the right lane points (assuming they are ordered)
                std::vector<cv::Point2f> rightPoints;
                rightPoints.push_back(cv::Point2f(rightCurve.front().x, rightCurve.front().y));
                rightPoints.push_back(cv::Point2f(rightCurve[rightCurve.size()/2].x, rightCurve[rightCurve.size()/2].y));
                rightPoints.push_back(cv::Point2f(rightCurve.back().x, rightCurve.back().y));

                std::vector<float> x_vals, y_vals;
                for (const auto &pt : rightPoints)
                {
                    x_vals.push_back(pt.x);
                    y_vals.push_back(pt.y);
                }
                cv::Mat x_mat(x_vals), y_mat(y_vals);
                cv::Mat rightCoeffs = polyfit(y_mat, x_mat, 2);
                if (!rightCoeffs.empty() && rightCoeffs.rows >= 3)
                {
                    // Shift the right lane left by laneWidthEstimate to predict the left lane.
                    double a = rightCoeffs.at<double>(0);
                    double b = rightCoeffs.at<double>(1);
                    double c = rightCoeffs.at<double>(2) - laneWidthEstimate;  // shift left
                    for (int y = height; y >= height / 3; y -= 5)
                    {
                        double x = a * y * y + b * y + c;
                        predictedLeftCurve.push_back(cv::Point(round(x), y));
                    }
                    leftCurve = predictedLeftCurve;
            
                }
            }
            else
            {
                // Fall back to using pure Kalman prediction if no measurement is available from either lane.
                std::vector<cv::Point> predictedLeftCurve;
                float bottom_x = leftPredicted.at<float>(0);
                float mid_x    = leftPredicted.at<float>(1);
                float top_x    = leftPredicted.at<float>(2);
                int bottom_y = height;
                int mid_y    = height / 2;
                int top_y    = height / 3;
                cv::Mat Y = (cv::Mat_<double>(3, 1) << bottom_y, mid_y, top_y);
                cv::Mat X = (cv::Mat_<double>(3, 1) << bottom_x, mid_x, top_x);
                cv::Mat coeffs = polyfit(Y, X, 2);
                if (!coeffs.empty() && coeffs.rows >= 3)
                {
                    for (int y = height; y >= height / 3; y -= 5)
                    {
                        double x = coeffs.at<double>(0) * y * y +
                                   coeffs.at<double>(1) * y +
                                   coeffs.at<double>(2);
                        predictedLeftCurve.push_back(cv::Point(round(x), y));
                    }
                    leftCurve = predictedLeftCurve;
                }
            }
        }

        // ----- Handle missing right lane -----
        if (!rightCurve.empty() && rightCurve.size() >= 3)
        {
            int bottom_idx = 0;
            int mid_idx    = rightCurve.size() / 2;
            int top_idx    = rightCurve.size() - 1;
            cv::Mat measurement = (cv::Mat_<float>(3, 1) << 
                                    rightCurve[bottom_idx].x,
                                    rightCurve[mid_idx].x,
                                    rightCurve[top_idx].x);
            rightLaneKF.correct(measurement);
        }
        else  // Right lane is lost or insufficient
        {
            std::vector<cv::Point> predictedRightCurve;
            int height = frame.rows;
            
            if (!leftCurve.empty() && leftCurve.size() >= 3)
            {
                // Use the good left lane to predict the missing right lane.
                std::vector<cv::Point2f> leftPoints;
                leftPoints.push_back(cv::Point2f(leftCurve.front().x, leftCurve.front().y));
                leftPoints.push_back(cv::Point2f(leftCurve[leftCurve.size()/2].x, leftCurve[leftCurve.size()/2].y));
                leftPoints.push_back(cv::Point2f(leftCurve.back().x, leftCurve.back().y));

                std::vector<float> x_vals, y_vals;
                for (const auto &pt : leftPoints)
                {
                    x_vals.push_back(pt.x);
                    y_vals.push_back(pt.y);
                }
                cv::Mat x_mat(x_vals), y_mat(y_vals);
                cv::Mat leftCoeffs = polyfit(y_mat, x_mat, 2);
                if (!leftCoeffs.empty() && leftCoeffs.rows >= 3)
                {
                    double a = leftCoeffs.at<double>(0);
                    double b = leftCoeffs.at<double>(1);
                    double c = leftCoeffs.at<double>(2);
                    double d = leftCoeffs.at<double>(3) + laneWidthEstimate;  // shift right
                    for (int y = height; y >= height / 3; y -= 5)
                    {
                        double x = a * y * y * y + b * y * y + c * y + d;
                        predictedRightCurve.push_back(cv::Point(round(x), y));
                    }
                    rightCurve = predictedRightCurve;
                }
            }
            else
            {
                // Fall back to using pure Kalman prediction.
                std::vector<cv::Point> predictedRightCurve;
                float bottom_x = rightPredicted.at<float>(0);
                float mid_x    = rightPredicted.at<float>(1);
                float top_x    = rightPredicted.at<float>(2);
                int bottom_y = height;
                int mid_y    = height / 2;
                int top_y    = height / 3;
                cv::Mat Y = (cv::Mat_<double>(3, 1) << bottom_y, mid_y, top_y);
                cv::Mat X = (cv::Mat_<double>(3, 1) << bottom_x, mid_x, top_x);
                cv::Mat coeffs = polyfit(Y, X, 2);
                if (!coeffs.empty() && coeffs.rows >= 3)
                {
                    for (int y = height; y >= height / 3; y -= 5)
                    {
                        double x = coeffs.at<double>(0) * y * y +
                                   coeffs.at<double>(1) * y +
                                   coeffs.at<double>(2);
                        predictedRightCurve.push_back(cv::Point(round(x), y));
                    }
                    rightCurve = predictedRightCurve;
                }
            }
        }
    }

    // Update lane width estimate when both curves are detected
    if (!leftCurve.empty() && !rightCurve.empty())
    {
        int bottomY = frame.rows - 1;
        int leftX = -1, rightX = -1;

        // Find points near the bottom of the image
        for (const auto& pt : leftCurve)
        {
            if (pt.y == bottomY || (leftX == -1 && pt.y > frame.rows * 0.7))
            {
                leftX = pt.x;
                break;
            }
        }

        for (const auto& pt : rightCurve)
        {
            if (pt.y == bottomY || (rightX == -1 && pt.y > frame.rows * 0.7))
            {
                rightX = pt.x;
                break;
            }
        }

        if (leftX != -1 && rightX != -1)
        {
            double currentWidth = rightX - leftX;
            // Exponential moving average to smooth lane width estimate
            laneWidthEstimate = 0.3 * currentWidth + 0.7 * laneWidthEstimate;
            // laneWidthEstimate = std::max(minLaneWidth, std::min(, laneWidthEstimate));
        }
    }

    double alpha = 0.3;
    if (!firstFrame)
    {
        // Apply moving average to left curve if it exists
        if (!leftCurve.empty())
        {
            // Add current curve to history
            leftLaneHistory.push_back(leftCurve);
            if (leftLaneHistory.size() > historySize)
            {
                leftLaneHistory.pop_front();
            }

            // Only apply averaging if we have enough history
            if (leftLaneHistory.size() >= 2)
            {
                // Create a copy of the current curve for averaging
                std::vector<cv::Point> averagedLeftCurve = leftCurve;

                // For each point in the curve
                for (size_t i = 0; i < leftCurve.size(); i++)
                {
                    int sumX = 0, sumY = 0;
                    double totalWeight = 0;

                    // Average across history (weighted, with recent frames
                    // having more weight)
                    for (size_t h = 0; h < leftLaneHistory.size(); h++)
                    {
                        if (i < leftLaneHistory[h].size())
                        {
                            // More recent frames get higher weight
                            double weight = (h + 1.0) / leftLaneHistory.size();
                            sumX += leftLaneHistory[h][i].x * weight;
                            sumY += leftLaneHistory[h][i].y * weight;
                            totalWeight += weight;
                        }
                    }

                    if (totalWeight > 0)
                    {
                        averagedLeftCurve[i].x =
                            static_cast<int>(sumX / totalWeight);
                        averagedLeftCurve[i].y =
                            static_cast<int>(sumY / totalWeight);
                    }
                }
                leftCurve = averagedLeftCurve;
            }

            // Dynamic alpha smoothing based on curvature
            if (!prevLeftCurve.empty() &&
                prevLeftCurve.size() == leftCurve.size())
            {
                double curvature = 0;
                if (leftCurve.size() >= 3)
                {
                    // Calculate curvature using all points
                    std::vector<float> x_vals, y_vals;
                    for (const auto& pt : leftCurve)
                    {
                        x_vals.push_back(pt.x);
                        y_vals.push_back(pt.y);
                    }

                    cv::Mat x_mat(x_vals), y_mat(y_vals);
                    cv::Mat coeffs = polyfit(y_mat, x_mat, 2);
                    if (!coeffs.empty() && coeffs.rows >= 3)
                    {
                        curvature = std::abs(coeffs.at<double>(0));
                    }
                }

                // Lower alpha for smoother transitions with higher curvature
                double dynamicAlpha = std::min(0.4, alpha + curvature * 1000);

                for (size_t i = 0; i < leftCurve.size(); i++)
                {
                    leftCurve[i].x = static_cast<int>(
                        dynamicAlpha * leftCurve[i].x +
                        (1 - dynamicAlpha) * prevLeftCurve[i].x);
                    leftCurve[i].y = static_cast<int>(
                        dynamicAlpha * leftCurve[i].y +
                        (1 - dynamicAlpha) * prevLeftCurve[i].y);
                }
            }
        }

        // Apply moving average to right curve if it exists
        if (!rightCurve.empty())
        {
            // Add current curve to history
            rightLaneHistory.push_back(rightCurve);
            if (rightLaneHistory.size() > historySize)
            {
                rightLaneHistory.pop_front();
            }

            // Only apply averaging if we have enough history
            if (rightLaneHistory.size() >= 2)
            {
                // Create a copy of the current curve for averaging
                std::vector<cv::Point> averagedRightCurve = rightCurve;

                // For each point in the curve
                for (size_t i = 0; i < rightCurve.size(); i++)
                {
                    int sumX = 0, sumY = 0;
                    double totalWeight = 0;

                    // Average across history (weighted, with recent frames
                    // having more weight)
                    for (size_t h = 0; h < rightLaneHistory.size(); h++)
                    {
                        if (i < rightLaneHistory[h].size())
                        {
                            // More recent frames get higher weight
                            double weight = (h + 1.0) / rightLaneHistory.size();
                            sumX += rightLaneHistory[h][i].x * weight;
                            sumY += rightLaneHistory[h][i].y * weight;
                            totalWeight += weight;
                        }
                    }

                    if (totalWeight > 0)
                    {
                        averagedRightCurve[i].x =
                            static_cast<int>(sumX / totalWeight);
                        averagedRightCurve[i].y =
                            static_cast<int>(sumY / totalWeight);
                    }
                }
                rightCurve = averagedRightCurve;
            }

            // Dynamic alpha smoothing based on curvature
            if (!prevRightCurve.empty() &&
                prevRightCurve.size() == rightCurve.size())
            {
                double curvature = 0;
                if (rightCurve.size() >= 3)
                {
                    // Calculate curvature using all points
                    std::vector<float> x_vals, y_vals;
                    for (const auto& pt : rightCurve)
                    {
                        x_vals.push_back(pt.x);
                        y_vals.push_back(pt.y);
                    }

                    cv::Mat x_mat(x_vals), y_mat(y_vals);
                    cv::Mat coeffs = polyfit(y_mat, x_mat, 2);
                    if (!coeffs.empty() && coeffs.rows >= 3)
                    {
                        curvature = std::abs(coeffs.at<double>(0));
                    }
                }

                // Lower alpha for smoother transitions with higher curvature
                double dynamicAlpha = std::min(0.4, alpha + curvature * 1000);

                for (size_t i = 0; i < rightCurve.size(); i++)
                {
                    rightCurve[i].x = static_cast<int>(
                        dynamicAlpha * rightCurve[i].x +
                        (1 - dynamicAlpha) * prevRightCurve[i].x);
                    rightCurve[i].y = static_cast<int>(
                        dynamicAlpha * rightCurve[i].y +
                        (1 - dynamicAlpha) * prevRightCurve[i].y);
                }
            }
        }
    }

    // Save current curves for next frame
    prevLeftCurve  = leftCurve;
    prevRightCurve = rightCurve;
    // std::cout << "Saved left and right curves" << std::endl;

    if (!leftCurve.empty() && leftCurve.size() >= 3 &&
    !rightCurve.empty() && rightCurve.size() >= 3)
    {
        prevLeftCurve  = leftCurve;
        prevRightCurve = rightCurve;
        // Also update your previous points used for clustering:
        prevLeftPoints = leftCurve;
        prevRightPoints = rightCurve;
    }
    else
    {
        // Add: Reset history after too many missed detections
        static int missedDetectionCount = 0;
        if (++missedDetectionCount > 10) {
            // Reset history after too many missed detections
            prevLeftCurve.clear();
            prevRightCurve.clear();
            missedDetectionCount = 0;
    }
    }

    // 8. Compute center lane as average of left and right lanes
    std::vector<cv::Point> midCurve;
    if (!leftCurve.empty() && !rightCurve.empty())
    {
        // Make sure we have equal length curves by resampling if needed
        int numPoints = std::min(leftCurve.size(), rightCurve.size());
        for (int i = 0; i < numPoints; i++)
        {
            size_t leftIdx  = i * leftCurve.size() / numPoints;
            size_t rightIdx = i * rightCurve.size() / numPoints;

            int midX = (leftCurve[leftIdx].x + rightCurve[rightIdx].x) / 2;
            int midY = (leftCurve[leftIdx].y + rightCurve[rightIdx].y) / 2;
            midCurve.push_back(cv::Point(midX, midY));
        }
    }

    // Limit the maximum curve drift from center
    if (!leftCurve.empty() && !rightCurve.empty()) {
        int width = frame.cols;
        float centerX = width / 2.0f;
        float maxOffsetDistance = width * 0.3f; // Maximum allowed offset (30% of frame width)
        
        // Calculate current lane midpoint at each y-level
        for (size_t i = 0; i < std::min(leftCurve.size(), rightCurve.size()); i++) {
            size_t leftIdx = i * leftCurve.size() / std::min(leftCurve.size(), rightCurve.size());
            size_t rightIdx = i * rightCurve.size() / std::min(leftCurve.size(), rightCurve.size());
            
            float midX = (leftCurve[leftIdx].x + rightCurve[rightIdx].x) / 2.0f;
            float offset = midX - centerX;
            
            // If offset exceeds limit, adjust both lane curves
            if (std::abs(offset) > maxOffsetDistance) {
                float adjustment = offset - (offset > 0 ? maxOffsetDistance : -maxOffsetDistance);
                
                // Apply adjustment to this point in both curves
                leftCurve[leftIdx].x -= adjustment;
                rightCurve[rightIdx].x -= adjustment;
            }
        }
        
    }

    // 9. Calculate reference point for lateral error - IMPROVED HANDLING
    cv::Point midPoint;
    int height = frame.rows;
    int width  = frame.cols;

    if (!midCurve.empty())
    {
        int targetY = height - (1 * height / 3); // 1/3 up from bottom

        // Find closest point to target Y
        size_t closestIdx = 0;
        int minDistance   = std::abs(midCurve[0].y - targetY);

        for (size_t i = 1; i < midCurve.size(); i++)
        {
            int distance = std::abs(midCurve[i].y - targetY);
            if (distance < minDistance)
            {
                minDistance = distance;
                closestIdx  = i;
            }
        }

        // Use the point at found index
        midPoint = midCurve[closestIdx];

        // Sanity check against convergence - verify lane separation at this
        // height
        bool lanesConvergedIncorrectly = false;

        if (!leftCurve.empty() && !rightCurve.empty())
        {
            // Get left and right lane positions at the same height
            cv::Point leftPos, rightPos;
            for (const auto& pt : leftCurve)
            {
                if (abs(pt.y - targetY) < 20)
                {
                    leftPos = pt;
                    break;
                }
            }

            for (const auto& pt : rightCurve)
            {
                if (abs(pt.y - targetY) < 20)
                {
                    rightPos = pt;
                    break;
                }
            }

            // Check if lanes are too close together
            if (leftPos.x != 0 && rightPos.x != 0)
            {
                float laneDistance = rightPos.x - leftPos.x;
                // If lanes are too close or crossed, something is wrong
                if (laneDistance < laneWidthEstimate * 0.7 ||
                    leftPos.x > rightPos.x)
                {
                    lanesConvergedIncorrectly = true;
                }
            }
        }

        if (prevMidPoint.x != -1 && prevMidPoint.y != -1)
        {
            // Use LOWER smoothing values for faster response
            float gamma =
                lanesConvergedIncorrectly ? 0.7 : 0.5; // Reduced from 0.9/0.8

            if (lanesConvergedIncorrectly)
            {
                float centerBias =
                    0.3; // Increased center bias when lanes converge
                midPoint.x = static_cast<int>(
                    gamma * prevMidPoint.x +
                    (1 - gamma) * ((1 - centerBias) * midPoint.x +
                                   centerBias * (width / 2)));
            }
            else
            {
                midPoint.x = static_cast<int>(gamma * prevMidPoint.x +
                                              (1 - gamma) * midPoint.x);
            }
            midPoint.y = static_cast<int>(gamma * prevMidPoint.y +
                                          (1 - gamma) * midPoint.y);
        }

        // Visual indicator of target Y
        cv::line(frame, cv::Point(0, targetY), cv::Point(width, targetY),
                 cv::Scalar(0, 255, 255), 1);
    }
    else
    {
        // IMPROVED FALLBACK: smarter handling when no midCurve is detected
        if (prevMidPoint.x != -1 && prevMidPoint.y != -1)
        {
            // Check if we're likely in a curve by examining previous lane
            // curves
            bool inCurve         = false;
            float curveDirection = 0.0f; // -1 = left curve, +1 = right curve

            // Use previous lane curves to determine if we're in a curve
            if (!prevLeftCurve.empty() && prevLeftCurve.size() >= 3)
            {
                // Sample 3 points from the curve to estimate curvature
                cv::Point top    = prevLeftCurve[prevLeftCurve.size() - 1];
                cv::Point middle = prevLeftCurve[prevLeftCurve.size() / 2];
                cv::Point bottom = prevLeftCurve[0];

                // Calculate basic curvature
                float dx1 = middle.x - bottom.x;
                float dx2 = top.x - middle.x;

                // If there's a significant change in direction, we're in a
                // curve
                if (abs(dx2 - dx1) > width * 0.03)
                {
                    inCurve        = true;
                    curveDirection = (dx2 > dx1) ? 1.0f : -1.0f;
                }
            }

            // Use right curve as backup if left doesn't show a curve
            if (!inCurve && !prevRightCurve.empty() &&
                prevRightCurve.size() >= 3)
            {
                cv::Point top    = prevRightCurve[prevRightCurve.size() - 1];
                cv::Point middle = prevRightCurve[prevRightCurve.size() / 2];
                cv::Point bottom = prevRightCurve[0];

                float dx1 = middle.x - bottom.x;
                float dx2 = top.x - middle.x;

                if (abs(dx2 - dx1) > width * 0.03)
                {
                    inCurve        = true;
                    curveDirection = (dx2 > dx1) ? 1.0f : -1.0f;
                }
            }

            // Different handling for curves vs. straight sections
            if (inCurve)
            {
                // In a curve: maintain curve trajectory with slower decay
                float curveDecay = 0.85f; // Slow decay in curves
                int errorReducedX =
                    width / 2 + (prevMidPoint.x - width / 2) * curveDecay;

                // Add a slight bias in the direction of the curve
                errorReducedX += curveDirection * width * 0.01;

                midPoint = cv::Point(errorReducedX, height * 2 / 3);
            }
            else
            {
                // Straight section: faster return to center
                float straightDecay = 0.7f; // Faster decay when straight
                int errorReducedX =
                    width / 2 + (prevMidPoint.x - width / 2) * straightDecay;
                midPoint = cv::Point(errorReducedX, height * 2 / 3);
            }
        }
        else
        {
            // No previous point - use center
            midPoint = cv::Point(width / 2, height * 2 / 3);
        }
    }

    // Update midPoint for next frame
    prevMidPoint = midPoint;
    prevMidCurve = midCurve;
    firstFrame   = false;

    // Calculate normalized lateral error (-1.0 to 1.0) with improved dampening
    float centerX  = width / 2;
    float rawError = (midPoint.x - centerX) / (width / 2.0f);

    // Apply rate limiting to error changes
    static float prevError       = 0.0f;
    const float MAX_ERROR_CHANGE = 0.3f; // Maximum allowed change per frame

    float errorChange = rawError - prevError;
    if (std::abs(errorChange) > MAX_ERROR_CHANGE)
    {
        errorChange = (errorChange > 0) ? MAX_ERROR_CHANGE : -MAX_ERROR_CHANGE;
    }

    float lateralError = prevError + errorChange;
    prevError          = lateralError;

    const float MAX_ERROR = 1.5f;
    if (lateralError > MAX_ERROR) {
        lateralError = MAX_ERROR;
        prevError = MAX_ERROR; // Update prevError as well
    } else if (lateralError < -MAX_ERROR) {
        lateralError = -MAX_ERROR;
        prevError = -MAX_ERROR; // Update prevError as well
    }
    laneError = lateralError;


    // Draw the final lane visualization
    drawLanes(frame, leftCurve, rightCurve);
    sendCoefs(leftCurve, rightCurve);
    // if (!this->leftCoeffs.empty() && this->leftCoeffs.rows >= 3) {
    //     std::cout << "Left lane coeffs: ["
    //               << this->leftCoeffs.at<double>(0) << ", "  // a (quadratic)
    //               << this->leftCoeffs.at<double>(1) << ", "  // b (linear)
    //               << this->leftCoeffs.at<double>(2) << "]"   // c (constant)
    //               << std::endl;
    // }
    // if (!this->rightCoeffs.empty() && this->rightCoeffs.rows >= 3) {
    //     std::cout << "Right lane coeffs: ["
    //               << this->rightCoeffs.at<double>(0) << ", "  // a (quadratic)
    //               << this->rightCoeffs.at<double>(1) << ", "  // b (linear)
    //               << this->rightCoeffs.at<double>(2) << "]"   // c (constant)
    //               << std::endl;
    // }

    // Draw center lane and reference point
    if (!midCurve.empty())
    {
        for (size_t i = 1; i < midCurve.size(); i++)
        {
            cv::line(frame, midCurve[i - 1], midCurve[i], cv::Scalar(0, 0, 255),
                     2);
        }
    }

    // Draw the reference point
    cv::circle(frame, midPoint, 8, cv::Scalar(255, 0, 255), -1);
    section_end = std::chrono::steady_clock::now();
    timings["3. Kalman filtering"] = std::chrono::duration_cast<std::chrono::microseconds>(section_end - section_start).count() / 1000.0;
    // Display lateral error as text
    std::string errorText = "Error: " + std::to_string(lateralError);
    cv::putText(frame, errorText, cv::Point(20, 90), cv::FONT_HERSHEY_SIMPLEX,
                0.7, cv::Scalar(255, 255, 255), 2);
    for (const auto& pair : timings) {
        std::cout << pair.first << ": " << pair.second << " ms" << std::endl;
    }

}

void LaneDetector::computeMidLanePolynomial()
{
    // Only compute if we have both lane polynomials
    if (!leftCoeffs.empty() && !rightCoeffs.empty() && 
        leftCoeffs.rows >= 4 && rightCoeffs.rows >= 4) {
        
        // Create a new Mat for mid coefficients
        midCoeffs = cv::Mat(4, 1, CV_64F);
        
        // Average each coefficient
        for (int i = 0; i < 4; i++) {
            midCoeffs.at<double>(i) = (leftCoeffs.at<double>(i) + rightCoeffs.at<double>(i)) / 2.0;
        }
        
        // Log the mid coefficients
        std::cout << "Mid lane coefficients: ["
                  << midCoeffs.at<double>(0) << ", "  // a (cubic)
                  << midCoeffs.at<double>(1) << ", "  // b (quadratic)
                  << midCoeffs.at<double>(2) << ", "  // c (linear)
                  << midCoeffs.at<double>(3) << "]"   // d (constant)
                  << std::endl;
    } else {
        // Clear mid coefficients if we don't have valid lane polynomials
        midCoeffs = cv::Mat();
    }
}

void LaneDetector::drawLanes(cv::Mat& frame,
                             const std::vector<cv::Point>& leftCurve,
                             const std::vector<cv::Point>& rightCurve)
{
    // Draw original points (that were used to calculate the lanes)
    for (const auto& pt : prevLeftPoints)
    {
        cv::circle(frame, pt, 3, cv::Scalar(125, 0, 0), -1); // Dark blue for
        // left points
    }

    for (const auto& pt : prevRightPoints)
    {
        cv::circle(frame, pt, 3, cv::Scalar(0, 125, 0), -1); // Dark green for
        // right points
    }
    // Draw left lane in blue
    if (!leftCurve.empty() && leftCurve.size() >= 2)
    {
        for (size_t i = 1; i < leftCurve.size(); i++)
        {
            cv::line(frame, leftCurve[i - 1], leftCurve[i],
                     cv::Scalar(255, 0, 0), 3);
        }
    }

    // Draw right lane in green
    if (!rightCurve.empty() && rightCurve.size() >= 2)
    {
        for (size_t i = 1; i < rightCurve.size(); i++)
        {
            cv::line(frame, rightCurve[i - 1], rightCurve[i],
                     cv::Scalar(0, 255, 0), 3);
        }
    }
}


// Helper: Filter points by ROI and density
static std::vector<cv::Point> filterPoints(const std::vector<cv::Point>& points, const cv::Mat& frame) {
    std::vector<cv::Point> filtered, densityFiltered;
    int width = frame.cols, height = frame.rows;
    
    // Define trapezoidal ROI
    // Bottom width: 80% of frame width (10% margin on each side)
    // Top width: 50% of frame width (25% margin on each side)
    // Bottom position: bottom of the frame
    // Top position: 15% from the top of the frame
    
    float bottomY = height;
    float topY = height * 0.30f;
    
    float bottomLeftX = width * - 1.5f;
    float bottomRightX = width * 2.5f;
    
    float topLeftX = width * 0.3f;
    float topRightX = width * 0.7f;
    
    // ROI filtering using trapezoidal shape
    for (const auto& pt : points) {
        // Calculate expected x boundaries at this y level using linear interpolation
        float yRatio = (pt.y - topY) / (bottomY - topY);
        
        // If point is above the ROI, skip it
        if (yRatio < 0) continue;
        
        float leftBoundary = topLeftX + yRatio * (bottomLeftX - topLeftX);
        float rightBoundary = topRightX + yRatio * (bottomRightX - topRightX);
        
        if (pt.y >= topY && pt.x >= leftBoundary && pt.x <= rightBoundary) {
            filtered.push_back(pt);
        }
    }
    
    // Visualize the trapezoid (optional)
    std::vector<cv::Point> trapezoid = {
        cv::Point(bottomLeftX, bottomY),
        cv::Point(bottomRightX, bottomY),
        cv::Point(topRightX, topY),
        cv::Point(topLeftX, topY)
    };
    
    cv::polylines(frame, std::vector<std::vector<cv::Point>>{trapezoid}, true, cv::Scalar(0, 255, 255), 2);
    
    // Density filtering: require at least 2 neighbors within a radius (reduced from 3)
    float radius = width * 0.03f;  // Increased from 0.025f
    for (size_t i = 0; i < filtered.size(); ++i) {
        int neighborCount = 0;
        for (size_t j = 0; j < filtered.size(); ++j) {
            if (i == j) continue;
            float dx = filtered[i].x - filtered[j].x;
            float dy = filtered[i].y - filtered[j].y;
            if (std::sqrt(dx * dx + dy * dy) < radius)
                neighborCount++;
        }
        if (neighborCount >= 2)  // Reduced from 3
            densityFiltered.push_back(filtered[i]);
    }
    
    return densityFiltered;
}

// Helper: Compute expected left/right boundaries using history and laneWidthEstimate.
// If history is available, use previous lane curves; otherwise, fall back to the simple midline.
static void computeExpectedBoundaries(const cv::Mat& frame, int midX, float laneWidthEstimate,
                                      const std::vector<cv::Point>& prevLeftCurve,
                                      const std::vector<cv::Point>& prevRightCurve,
                                      int &expectedLeftBoundary, int &expectedRightBoundary)
{
    //int width = frame.cols;
    (void) frame;
    if (!prevLeftCurve.empty() && !prevRightCurve.empty()) {
        // Use the bottom-most points from previous curves.
        int leftX = prevLeftCurve.front().x;
        int rightX = prevRightCurve.front().x;
        int historyMidX = (leftX + rightX) / 2;
        expectedLeftBoundary  = historyMidX - static_cast<int>(laneWidthEstimate * 0.5f);
        expectedRightBoundary = historyMidX + static_cast<int>(laneWidthEstimate * 0.5f);
    } else {
        expectedLeftBoundary  = midX - static_cast<int>(laneWidthEstimate * 0.5f);
        expectedRightBoundary = midX + static_cast<int>(laneWidthEstimate * 0.5f);
    }
}

// Helper: Assign filtered points to left/right clusters using expected boundaries and an adaptive tolerance.
static void assignPointsToLanes(const std::vector<cv::Point>& points,
                                std::vector<cv::Point>& leftPoints,
                                std::vector<cv::Point>& rightPoints,
                                int expectedLeftBoundary,
                                int expectedRightBoundary,
                                float tolerance,
                                int midX)
{
    std::vector<cv::Point> ambiguous;
    for (const auto& pt : points) {
        if (pt.x < expectedLeftBoundary + tolerance)
            leftPoints.push_back(pt);
        else if (pt.x > expectedRightBoundary - tolerance)
            rightPoints.push_back(pt);
        else
            ambiguous.push_back(pt);
    }
    // For ambiguous points, decide based on closeness.
    for (const auto& pt : ambiguous) {
        float dLeft = std::abs(pt.x - expectedLeftBoundary);
        float dRight = std::abs(pt.x - expectedRightBoundary);
        if (dLeft < dRight && dLeft < tolerance)
            leftPoints.push_back(pt);
        else if (dRight < dLeft && dRight < tolerance)
            rightPoints.push_back(pt);
        else {
            // Fallback: use midline separation.
            if (pt.x < midX) leftPoints.push_back(pt);
            else rightPoints.push_back(pt);
        }
    }
        // Draw the current clustered points first (before curve fitting)
}


void LaneDetector::clusterLanePoints(const std::vector<cv::Point>& points,
    std::vector<cv::Point>& leftPoints,
    std::vector<cv::Point>& rightPoints,
    cv::Mat& frame)
{
    leftPoints.clear();
    rightPoints.clear();
    
    if (points.empty()) return;
    
    // When we have previous lanes, use them as seeds for clustering
    if (!prevLeftCurve.empty() && !prevRightCurve.empty()) {
        // Pre-compute left and right lane polynomials
        std::vector<float> left_x, left_y, right_x, right_y;
        for (const auto& pt : prevLeftCurve) {
            left_x.push_back(pt.x);
            left_y.push_back(pt.y);
        }
        
        for (const auto& pt : prevRightCurve) {
            right_x.push_back(pt.x);
            right_y.push_back(pt.y);
        }
        
        cv::Mat leftX(left_x), leftY(left_y), rightX(right_x), rightY(right_y);
        cv::Mat leftCoeffs = polyfit(leftY, leftX, 3);
        cv::Mat rightCoeffs = polyfit(rightY, rightX, 3);
        
        // If polynomials are valid
        if (!leftCoeffs.empty() && !rightCoeffs.empty() && 
            leftCoeffs.rows >= 3 && rightCoeffs.rows >= 3) {
            
            // Calculate adaptive threshold based on lane width
            double laneWidth = laneWidthEstimate;
            double threshold = laneWidth * 0.4; // 40% of lane width
            
            // Assign points to left or right based on distance to curves
            for (const auto& pt : points) {
                // Compute expected x positions on both curves at this y
                double leftX = leftCoeffs.at<double>(0) * pt.y * pt.y + 
                             leftCoeffs.at<double>(1) * pt.y + 
                             leftCoeffs.at<double>(2);
                             
                double rightX = rightCoeffs.at<double>(0) * pt.y * pt.y + 
                              rightCoeffs.at<double>(1) * pt.y + 
                              rightCoeffs.at<double>(2);
                
                // Calculate distances to both curves
                double leftDist = std::abs(pt.x - leftX);
                double rightDist = std::abs(pt.x - rightX);
                
                // Assign to closest curve if within threshold
                if (leftDist < rightDist && leftDist < threshold) {
                    leftPoints.push_back(pt);
                } else if (rightDist < leftDist && rightDist < threshold) {
                    rightPoints.push_back(pt);
                }
                // Points not close to either curve are ignored
            }
            
            return;
        }
    }
    
    // Fallback: use simple x-based clustering
    int width = 0;
    for (const auto& pt : points) {
        width = std::max(width, pt.x);
    }
    
    int midX = width / 2;
    for (const auto& pt : points) {
        if (pt.x < midX) {
            leftPoints.push_back(pt);
        } else {
            rightPoints.push_back(pt);
        }
    }
}


std::vector<cv::Point>
LaneDetector::fitCurveToPoints(const std::vector<cv::Point>& points,
                               cv::Mat& frame)
{
    if (points.size() < 4)
    {
        return std::vector<cv::Point>();
    }

    // Step 1: Calculate local density for each point
    std::vector<float> weights(points.size(), 1.0f);
    float densityRadius = frame.cols * 0.03f; // 3% of frame width as density measurement radius
    
    // Calculate point density weights
    for (size_t i = 0; i < points.size(); i++)
    {
        int neighborCount = 0;
        for (size_t j = 0; j < points.size(); j++)
        {
            if (i != j)
            {
                float dist = std::sqrt(std::pow(points[i].x - points[j].x, 2) + 
                                     std::pow(points[i].y - points[j].y, 2));
                if (dist < densityRadius)
                {
                    neighborCount++;
                    // Optionally give more weight to very close points
                    weights[i] += (1.0f - dist/densityRadius);
                }
            }
        }
        // Normalize weights to avoid extreme values
        weights[i] = std::min(5.0f, 1.0f + neighborCount * 0.5f);
    }

    // Step 2: Create weighted points for polynomial fitting
    std::vector<float> x_vals, y_vals;
    std::vector<float> weighted_x, weighted_y;
    
    for (size_t i = 0; i < points.size(); i++)
    {
        // Add each point multiple times based on its weight
        int repetitions = std::round(weights[i]);
        for (int r = 0; r < repetitions; r++)
        {
            x_vals.push_back(points[i].x);
            y_vals.push_back(points[i].y);
        }
        
        // Also keep a copy of the original values for debugging
        weighted_x.push_back(points[i].x);
        weighted_y.push_back(points[i].y);
    }

    // Step 3: Perform polynomial fitting with weighted points
    cv::Mat x_mat(x_vals), y_mat(y_vals);
    int degree = 3; // Quadratic curve
    cv::Mat coeffs = polyfit(y_mat, x_mat, degree);

    // Generate smooth curve
    std::vector<cv::Point> curve;
    if (!coeffs.empty() && coeffs.rows >= 4)
    {
        for (int y = frame.rows; y >= frame.rows * 0.35; y -= 5)
        {
            double x = coeffs.at<double>(0) * y * y * y +  // Cubic term
                    coeffs.at<double>(1) * y * y +      // Quadratic term
                    coeffs.at<double>(2) * y +          // Linear term
                    coeffs.at<double>(3);               // Constant term
            if (!std::isnan(x) && !std::isinf(x))
            {
                curve.push_back(cv::Point(round(x), y));
            }
        }
    }
    
    // Optional: Visualize the weights for debugging (comment out in production)
    /*
    for (size_t i = 0; i < points.size(); i++) {
        int radius = std::max(2, std::min(8, int(weights[i] * 1.5)));
        cv::circle(frame, points[i], radius, cv::Scalar(0, 0, 255), 1);
    }
    */
    
    return curve;
}


void LaneDetector::initKalmanFilters(const std::vector<cv::Point>& leftCurve,
                                     const std::vector<cv::Point>& rightCurve)
{
    // Track 3 key points (bottom, middle, top) of each lane
    // State: [x_bottom, x_middle, x_top, vx_bottom, vx_middle, vx_top]
    int stateSize   = 6;
    int measSize    = 3; // We measure x positions at three points
    int controlSize = 0; // No control input

    leftLaneKF.init(stateSize, measSize, controlSize, CV_32F);
    rightLaneKF.init(stateSize, measSize, controlSize, CV_32F);

    // Initialize state transition matrix (constant velocity model)
    cv::setIdentity(leftLaneKF.transitionMatrix);
    cv::setIdentity(rightLaneKF.transitionMatrix);
    for (int i = 0; i < 3; i++)
    {
        leftLaneKF.transitionMatrix.at<float>(i, i + 3)  = 1.0f;
        rightLaneKF.transitionMatrix.at<float>(i, i + 3) = 1.0f;
    }

    // Initialize measurement matrix
    cv::setIdentity(leftLaneKF.measurementMatrix, cv::Scalar(1));
    cv::setIdentity(rightLaneKF.measurementMatrix, cv::Scalar(1));

    // Set process noise covariance
    cv::setIdentity(leftLaneKF.processNoiseCov, cv::Scalar(1e-4));
    cv::setIdentity(rightLaneKF.processNoiseCov, cv::Scalar(1e-4));

    // Set measurement noise covariance
    cv::setIdentity(leftLaneKF.measurementNoiseCov, cv::Scalar(1e-1));
    cv::setIdentity(rightLaneKF.measurementNoiseCov, cv::Scalar(1e-1));

    // Set error covariance
    cv::setIdentity(leftLaneKF.errorCovPost, cv::Scalar(1));
    cv::setIdentity(rightLaneKF.errorCovPost, cv::Scalar(1));

    // Initialize state with current lane positions
    if (!leftCurve.empty() && leftCurve.size() >= 3)
    {
        int bottom_idx = 0;
        int mid_idx    = leftCurve.size() / 2;
        int top_idx    = leftCurve.size() - 1;

        leftLaneKF.statePost.at<float>(0) = leftCurve[bottom_idx].x;
        leftLaneKF.statePost.at<float>(1) = leftCurve[mid_idx].x;
        leftLaneKF.statePost.at<float>(2) = leftCurve[top_idx].x;
        // Initialize velocities as 0
        leftLaneKF.statePost.at<float>(3) = 0;
        leftLaneKF.statePost.at<float>(4) = 0;
        leftLaneKF.statePost.at<float>(5) = 0;
    }

    if (!rightCurve.empty() && rightCurve.size() >= 3)
    {
        int bottom_idx = 0;
        int mid_idx    = rightCurve.size() / 2;
        int top_idx    = rightCurve.size() - 1;

        rightLaneKF.statePost.at<float>(0) = rightCurve[bottom_idx].x;
        rightLaneKF.statePost.at<float>(1) = rightCurve[mid_idx].x;
        rightLaneKF.statePost.at<float>(2) = rightCurve[top_idx].x;
        // Initialize velocities as 0
        rightLaneKF.statePost.at<float>(3) = 0;
        rightLaneKF.statePost.at<float>(4) = 0;
        rightLaneKF.statePost.at<float>(5) = 0;
    }

    kfInitialized = true;
}

void LaneDetector::sendCoefs(const std::vector<cv::Point>& leftCurve,
                               const std::vector<cv::Point>& rightCurve)
{
    // Extract coefficients for the left curve
    if (!leftCurve.empty() && leftCurve.size() >= 3)
    {
        std::vector<float> x_vals, y_vals;
        for (const auto& pt : leftCurve)
        {
            x_vals.push_back(pt.x);
            y_vals.push_back(pt.y);
        }

        cv::Mat x_mat(x_vals), y_mat(y_vals);
        this->leftCoeffs = polyfit(y_mat, x_mat, 3);
    }

    // Extract coefficients for the right curve
    if (!rightCurve.empty() && rightCurve.size() >= 3)
    {
        std::vector<float> x_vals, y_vals;
        for (const auto& pt : rightCurve)
        {
            x_vals.push_back(pt.x);
            y_vals.push_back(pt.y);
        }

        cv::Mat x_mat(x_vals), y_mat(y_vals);
        rightCoeffs = polyfit(y_mat, x_mat, 3);
    }
    computeMidLanePolynomial();

}


void LaneDetector::visualizeBothViews(cv::Mat& display_frame)
{
    // Create a copy of the original frame for visualization
    cv::Mat originalWithLanes = original_frame.clone();
    
    // Transform lane curves back to original perspective for visualization
    if (!prevLeftCurve.empty() && prevLeftCurve.size() >= 2) {
        std::vector<cv::Point2f> bevPoints, origPoints;
        
        // Convert to Point2f for transformation
        for (const auto& pt : prevLeftCurve) {
            bevPoints.push_back(cv::Point2f(pt.x, pt.y));
        }
        
        // Transform points using inverse IPM
        cv::perspectiveTransform(bevPoints, origPoints, ipm.getInvPerspectiveMatrix());
        
        // Draw transformed lines
        for (size_t i = 1; i < origPoints.size(); i++) {
            cv::line(originalWithLanes, origPoints[i-1], origPoints[i], cv::Scalar(255, 0, 0), 3);
        }
    }
    
    if (!prevRightCurve.empty() && prevRightCurve.size() >= 2) {
        std::vector<cv::Point2f> bevPoints, origPoints;
        
        for (const auto& pt : prevRightCurve) {
            bevPoints.push_back(cv::Point2f(pt.x, pt.y));
        }
        
        cv::perspectiveTransform(bevPoints, origPoints, ipm.getInvPerspectiveMatrix());
        
        for (size_t i = 1; i < origPoints.size(); i++) {
            cv::line(originalWithLanes, origPoints[i-1], origPoints[i], cv::Scalar(0, 255, 0), 3);
        }
    }
    
    // Resize both frames to same height if needed
    cv::Mat resizedOrig, resizedBEV;
    int targetHeight = display_frame.rows / 2;
    
    cv::resize(originalWithLanes, resizedOrig, cv::Size(targetHeight * originalWithLanes.cols / originalWithLanes.rows, targetHeight));
    cv::resize(display_frame, resizedBEV, cv::Size(targetHeight * display_frame.cols / display_frame.rows, targetHeight));
    
    // Create combined visualization
    cv::Mat combined;
    cv::hconcat(resizedOrig, resizedBEV, combined);
    
    // Add labels
    cv::putText(combined, "Original View", cv::Point(20, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(combined, "Bird's Eye View", cv::Point(resizedOrig.cols + 20, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    // Replace the display frame with the combined view
    display_frame = combined.clone();
}