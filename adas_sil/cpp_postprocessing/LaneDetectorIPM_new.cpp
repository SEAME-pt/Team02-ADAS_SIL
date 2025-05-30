#include "LaneDetectorIPM_new.hpp"

#include <iostream>

using namespace cv;
using namespace std;

LaneDetector::LaneDetector()
    :laneWidthEstimate(0.0), firstFrame(true), frame_count(0)
{
    // set kalman filter status to false
    inputData = new float[3 * HEIGHT * WIDTH];
    outputData = new float[1 * HEIGHT * WIDTH];

    std::memset(inputData, 0, 3 * HEIGHT * WIDTH * sizeof(float));
    std::memset(outputData, 0, 1 * HEIGHT * WIDTH * sizeof(float));

    leftCoeffs = cv::Mat::zeros(4, 1, CV_64F);
    rightCoeffs = cv::Mat::zeros(4, 1, CV_64F);
    midCoeffs = cv::Mat::zeros(4, 1, CV_64F);
    std::cout << "LaneDetector initialized with inputData and outputData buffers." << std::endl;

    try {
        this->kalmanFilter = new ::KalmanFilter();
        this->kalmanFilter->init(); 
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error initializing KalmanFilter" << e.what() << std::endl;
    }

    try     
    {
        float cameraHeight = 1.5f;       // meters
        float cameraPitch = 15.0f;       // degrees down from horizontal
        float horizontalFOV = 105.0f;   // degrees
        float img_height = static_cast<float>(HEIGHT);
        float img_width = static_cast<float>(WIDTH);
        float h_fov_rad = horizontalFOV * CV_PI / 180.0f;
        float verticalFOV = 2.0f * std::atan((img_height/img_width) * std::tan(h_fov_rad/2.0f)) * 180.0f / CV_PI;
        float nearDistance = 1.5f;       // meters
        float farDistance = 15.0f;       // meters
        float laneWidth = 7.0f;          // meters
        bevSize = cv::Size(WIDTH, WIDTH);
        cv::Size origSize = cv::Size(WIDTH, HEIGHT);

        this->ipm     = new IPM();
        this->ipm->init(origSize, bevSize);
        this->ipm->calibrateFromCamera(cameraHeight, cameraPitch, horizontalFOV, verticalFOV,
                            nearDistance, farDistance, laneWidth);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error initializing IPM" << e.what() << std::endl;
    }
}

LaneDetector::~LaneDetector()
{
    delete[] inputData;
    delete[] outputData;
    delete kalmanFilter;
    delete ipm;
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

    cv::Mat bev_mask = ipm->applyIPM(mask);
    // cv::Mat bev_mask = ipm->transformPoints(mask);
    // Create a copy of the mask for ROI application
    cv::Mat roiMask = bev_mask.clone();
    std::vector<cv::Point> points;
   
    // Collect points from the ROI-filtered mask
    std::vector<cv::Point> maskPoints;
    cv::findNonZero(bev_mask, maskPoints);

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

    // Create a proper BEV image
    this->bev_image = cv::Mat(bevSize, CV_8UC3, cv::Scalar(0, 0, 0));
    // Draw detected points on BEV image
    for (const auto& pt : lanePoints) {
        cv::circle(bev_image, pt, 1, cv::Scalar(255, 255, 255), -1);
    }
    createLanesIPM(lanePoints, bev_image);
    
}

void LaneDetector::createLanesIPM(std::vector<cv::Point> lanePoints,
    cv::Mat& frame)
{

    std::vector<cv::Point> leftCurve;
    std::vector<cv::Point> rightCurve;
    std::vector<cv::Point> midCurve;
    std::vector<std::vector<cv::Point>> lanePolylines;

    currentFrame++;
    allPolylinesViz_ = frame.clone();
    frameWidth_ = frame.cols;
    frameHeight_ = frame.rows;

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
 
    lanePolylines = clusterLaneMask(laneMask, 30, 40, 6);

    drawPolyLanes(lanePolylines);

    
    section_start = std::chrono::steady_clock::now();
    float maxHorizontalDistance = frameWidth_ * 0.15; // 15% of frame width
    float maxVerticalGap = frameHeight_ * 0.35;       // 35% of frame height
    mergeLaneComponents(lanePolylines, maxHorizontalDistance, maxVerticalGap);

    
    section_end = std::chrono::steady_clock::now();
    timings["1. Binary mask"] = std::chrono::duration_cast<std::chrono::microseconds>(section_end - section_start).count() / 1000.0;
    
    section_start = std::chrono::steady_clock::now();
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
        

        kalmanFilter->updateLeftLaneFilter(leftCurve);
        kalmanFilter->updateRightLaneFilter(rightCurve);
        std::cout << "LALA 2" << std::endl;

        defineTrajectoryCurve(midCurve, leftCurve, rightCurve);
        kalmanFilter->updateMiddleLaneFilter(midCurve);
        std::cout << "LALA 3" << std::endl;

        prevLeftCurve = leftCurve;
        prevRightCurve = rightCurve;
        
        leftLaneLastUpdatedFrame = currentFrame;
        rightLaneLastUpdatedFrame = currentFrame;

    } else if (lanePolylines.size() == 1) {
        bool isLeftLane = checkIfLeftLane(lanePolylines);
        if (isLeftLane) {
            leftCurve = lanePolylines[0];

            prevLeftCurve = leftCurve;
            leftLaneLastUpdatedFrame = currentFrame;

            rightCurve = kalmanFilter->predictRightLaneCurve(frameHeight_, frameWidth_);
            checkPredicedCurve(rightCurve, leftCurve, true);

            defineTrajectoryCurve(midCurve, leftCurve, rightCurve);
            kalmanFilter->updateMiddleLaneFilter(midCurve);
        } else {

            rightCurve = lanePolylines[0];

            prevRightCurve = rightCurve;
            rightLaneLastUpdatedFrame = currentFrame;

            leftCurve = kalmanFilter->predictLeftLaneCurve(frameHeight_, frameWidth_);

            checkPredicedCurve(leftCurve, rightCurve, false);
            
            defineTrajectoryCurve(midCurve, leftCurve, rightCurve);
            kalmanFilter->updateMiddleLaneFilter(midCurve);
        }
        
        std::string statusMsg = isLeftLane ? "Using kalmanFilter RIGHT lane" : "Using kalmanFilter LEFT lane"; 
        cv::putText(allPolylinesViz_, statusMsg, cv::Point(20, 60), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    } else {  // No lanes detected


    if (kalmanFilter->midLaneInitialized) {
        // Only use Kalman prediction if initialized
        midCurve = kalmanFilter->predictMiddleLaneCurve(frameHeight_, frameWidth_);
    } else {
        // Create a default mid lane for cold start
        midCurve.clear();
        int centerX = frameWidth_ / 2;
        
        // Create points from bottom to top
        for (int y = frameHeight_ - 10; y >= 10; y -= frameHeight_/10) {
            midCurve.push_back(cv::Point(centerX, y));
        }
        
        // Add warning about using default lane
        cv::putText(allPolylinesViz_, "DEFAULT CENTER LANE", 
                  cv::Point(20, 120), cv::FONT_HERSHEY_SIMPLEX, 
                  0.7, cv::Scalar(0, 0, 255), 2);
        }

        // Draw the mid curve
        cv::Scalar midCurveColor = cv::Scalar(255, 255, 255); // White
        for (size_t i = 1; i < midCurve.size(); i++) {
            cv::line(allPolylinesViz_, midCurve[i-1], midCurve[i], midCurveColor, 3);
        }
    }

    midCoeffs = kalmanFilter->middleLaneCoeffs;
    leftCoeffs = kalmanFilter->leftLaneCoeffs;
    rightCoeffs = kalmanFilter->rightLaneCoeffs;
    createMidPointError(midCurve, frame);
}

std::vector<std::vector<cv::Point>> LaneDetector::clusterLaneMask(const cv::Mat& laneMask, int kernelSize, int minArea, int maxLanes) {    
    static cv::Mat verticalKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize * 3));
    static cv::Mat horizontalKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    
    cv::Mat result = laneMask.clone();
    cv::morphologyEx(result, result, cv::MORPH_CLOSE, verticalKernel);
    cv::morphologyEx(result, result, cv::MORPH_CLOSE, horizontalKernel);

    
    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(result, labels, stats, centroids, 8, CV_32S);
    

    std::vector<std::pair<int, float>> validComponents;
    validComponents.reserve(std::min(numLabels, maxLanes + 3));
    

    for (int i = 1; i < numLabels; i++) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > minArea) {
            float centerX = centroids.at<double>(i, 0);
            validComponents.push_back(std::make_pair(i, centerX));
        }
    }
    
    // Use partial sort instead of full sort when number of valid components bigger than maxLanes
    if (validComponents.size() > static_cast<size_t>(maxLanes)) {
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
    
    std::cout << "Found " << validComponents.size() << " valid lane components." << std::endl;
    // Reserve capacity for output
    std::vector<std::vector<cv::Point>> lanePolylines;
    lanePolylines.reserve(validComponents.size());
    

    std::cout << "Processing " << validComponents.size() << " valid lane components." << std::endl;
    // Process each lane with optimized extraction
    for (const auto& comp : validComponents) {
        int compIdx = comp.first;
        
        // Extract points more efficiently using row pointers
        std::vector<cv::Point> lanePoints;
        lanePoints.reserve(labels.rows/5);
        
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
            lanePolylines.push_back(std::move(lanePoints));
        }
    }
    std::cout << "Extracted " << lanePolylines.size() << " lane polylines." << std::endl;
    return lanePolylines;
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
    
    // cv::polylines(frame, std::vector<std::vector<cv::Point>>{trapezoid}, true, cv::Scalar(0, 255, 255), 2);
    
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
        cv::perspectiveTransform(bevPoints, origPoints, ipm->getInvPerspectiveMatrix());
        
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
        
        cv::perspectiveTransform(bevPoints, origPoints, ipm->getInvPerspectiveMatrix());
        
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

bool LaneDetector::validateLaneSeparation(const std::vector<std::vector<cv::Point>>& lanePolylines, float minLaneWidth) {
    if (lanePolylines.size() < 2) return true;
    
    // Calculate average distance between the two lanes
    float avgDistance = calculateLaneDistance(lanePolylines[0], lanePolylines[1]);
    
    // If lanes are too close, they're likely the same lane detected twice
    return avgDistance >= minLaneWidth;
}

void LaneDetector::checkPredicedCurve(std::vector<cv::Point>& predictedCurve, const std::vector<cv::Point>& realLane, bool isLeftLane) {
    float avgMiddleX = 0;
    float avgDetectedX = 0;
    float expectedHalfWidth = frameWidth_ * 0.50f; // Approximately lane width
    float expectedMiddleX = 0;

    // Calculate average X positions
    for (const auto& pt : predictedCurve) {
        avgMiddleX += pt.x;
    }
    avgMiddleX /= predictedCurve.size();

    for (const auto& pt : realLane) {
        avgDetectedX += pt.x;
    }
    avgDetectedX /= realLane.size();

    if (isLeftLane) {
        expectedMiddleX = avgDetectedX + expectedHalfWidth;
    } else {
        expectedMiddleX = avgDetectedX - expectedHalfWidth;
    }
    
    float error = std::abs(avgMiddleX - expectedMiddleX);
    
    if (error > frameWidth_ * 0.05f) {
        cv::putText(allPolylinesViz_, "Invalid curve prediction - using offset", 
                  cv::Point(20, 160), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        
        predictedCurve.clear();
        predictedCurve.reserve(realLane.size());
        if (isLeftLane) {
            for (const auto& pt : realLane) {
                predictedCurve.push_back(cv::Point(pt.x + expectedHalfWidth, pt.y));
            }
        } else {
            for (const auto& pt : realLane) {
                predictedCurve.push_back(cv::Point(pt.x - expectedHalfWidth, pt.y));
            }
        }
        
        if (isLeftLane) {
            kalmanFilter->updateRightLaneFilter(predictedCurve);
        } else {
            kalmanFilter->updateLeftLaneFilter(predictedCurve);
        }
    }
}


void LaneDetector::drawPolyLanes(std::vector<std::vector<cv::Point>> lanePolylines) {
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
            cv::line(allPolylinesViz_, lanePolylines[i][j-1], lanePolylines[i][j], color, 2);
        }
        
        // Add a label for each polyline
        if (!lanePolylines[i].empty()) {
            std::string label = "Lane " + std::to_string(i+1);
            cv::putText(allPolylinesViz_, label, lanePolylines[i][0], 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
        }
    }
    
    // Display the number of polylines found
    std::string countText = "Polylines: " + std::to_string(lanePolylines.size());
    cv::putText(allPolylinesViz_, countText, cv::Point(20, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
}


float LaneDetector::calculateLaneDistance(const std::vector<cv::Point>& lane1, 
                                        const std::vector<cv::Point>& lane2) {
    // Create normalized Y-position mapping of lane points
    std::map<int, cv::Point> lane1Points;
    std::map<int, cv::Point> lane2Points;
    
    // Normalize Y values to 0-100 range
    for (const auto& pt : lane1) {
        int normY = (pt.y * 100) / 480;  // Assuming 480 is max height
        lane1Points[normY] = pt;
    }
    
    for (const auto& pt : lane2) {
        int normY = (pt.y * 100) / 480;  // Assuming 480 is max height
        lane2Points[normY] = pt;
    }
    
    // Calculate average distance between lanes at matching Y positions
    float totalDist = 0;
    int matchCount = 0;
    
    for (const auto& p1 : lane1Points) {
        int y = p1.first;
        if (lane2Points.find(y) != lane2Points.end()) {
            // Calculate Euclidean distance
            float dist = cv::norm(p1.second - lane2Points[y]);
            totalDist += dist;
            matchCount++;
        }
    }
    
    return (matchCount > 0) ? (totalDist / matchCount) : FLT_MAX;
}

void LaneDetector::defineTrajectoryCurve(std::vector<cv::Point>& midCurve, std::vector<cv::Point>& leftCurve, std::vector<cv::Point>& rightCurve) {
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

    // Draw middle lane curve with white color and thicker line
    cv::Scalar midCurveColor = cv::Scalar(255, 255, 255); // White
    for (size_t i = 1; i < midCurve.size(); i++) {
        cv::line(allPolylinesViz_, midCurve[i-1], midCurve[i], midCurveColor, 3);
    }
}


void LaneDetector::createMidPointError(std::vector<cv::Point>& midCurve, cv::Mat frame) {
    cv::Point midPoint;
    int height = frame.rows;
    int width  = frame.cols;

    if (!midCurve.empty()) {
        int targetY = height - (1.5 * height / 3); // 1/3 up from bottom

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

        cv::circle(allPolylinesViz_, midPoint, 8, cv::Scalar(255, 0, 255), -1);

        float centerX  = width / 2;
        float rawError = (midPoint.x - centerX) / (width / 2.0f);

        // Apply rate limiting to error changes
        static float prevError       = 0.0f;
        const float MAX_ERROR_CHANGE = 1.0f; // Maximum allowed change per frame

        float errorChange = rawError - prevError;
        if (std::abs(errorChange) > MAX_ERROR_CHANGE)
        {
            errorChange = (errorChange > 0) ? MAX_ERROR_CHANGE : -MAX_ERROR_CHANGE;
        }

        float lateralError = prevError + errorChange;
        prevError          = lateralError;

        const float MAX_ERROR = 3.0f;
        if (lateralError > MAX_ERROR) {
            lateralError = MAX_ERROR;
            prevError = MAX_ERROR; // Update prevError as well
        } else if (lateralError < -MAX_ERROR) {
            lateralError = -MAX_ERROR;
            prevError = -MAX_ERROR; // Update prevError as well
        }

        laneError = lateralError;
    }

}

bool LaneDetector::checkIfLeftLane(const std::vector<std::vector<cv::Point>> &lanePolylines) {
    cv::Point lowestPoint(-1, -1);
    int centerX = frameWidth_ / 2;
    float avgX = 0;
    
    // Find lowest point and average X position
    for (const auto& pt : lanePolylines[0]) {
        if (pt.y > lowestPoint.y) {
            lowestPoint = pt;
        }
        avgX += pt.x;
    }
    avgX /= lanePolylines[0].size();
    
    // Draw detected lane's lowest point
    cv::circle(allPolylinesViz_, lowestPoint, 8, cv::Scalar(255, 0, 255), -1);
    
    bool hasValidLeftMemory = (currentFrame - leftLaneLastUpdatedFrame) < MAX_LANE_MEMORY_FRAMES;
    bool hasValidRightMemory = (currentFrame - rightLaneLastUpdatedFrame) < MAX_LANE_MEMORY_FRAMES;
    bool isLeftLane;

    // If we have previous lanes, use them to identify current lane
    if (hasValidLeftMemory || hasValidRightMemory) {
        
        float leftDistance = hasValidLeftMemory ? 
                    calculateLaneDistance(lanePolylines[0], prevLeftCurve) : 
                    FLT_MAX;

        float rightDistance = hasValidRightMemory ? 
                            calculateLaneDistance(lanePolylines[0], prevRightCurve) : 
                            FLT_MAX;

        if (hasValidLeftMemory) {
            float leftStaleness = 1.0f + 0.05f * (currentFrame - leftLaneLastUpdatedFrame);
            leftDistance *= leftStaleness;
        }
        
        if (hasValidRightMemory) {
            float rightStaleness = 1.0f + 0.05f * (currentFrame - rightLaneLastUpdatedFrame);
            rightDistance *= rightStaleness;
        }
        
        isLeftLane = leftDistance < rightDistance;
        
        std::string debugMsg = "Memory match: " + std::string(isLeftLane ? "LEFT" : "RIGHT") + 
                                " (L:" + std::to_string(leftDistance).substr(0,5) + 
                                "/R:" + std::to_string(rightDistance).substr(0,5) + ")";
        cv::putText(allPolylinesViz_, debugMsg, cv::Point(20, 80), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
        
        std::string staleMsg = "Staleness - L:" + std::to_string(currentFrame - leftLaneLastUpdatedFrame) +
                                " R:" + std::to_string(currentFrame - rightLaneLastUpdatedFrame);
        cv::putText(allPolylinesViz_, staleMsg, cv::Point(20, 100), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
    } else {
        // Fallback to position-based detection
        isLeftLane = avgX < centerX;
        
        if (!hasValidLeftMemory || !hasValidRightMemory) {
            cv::putText(allPolylinesViz_, "Memory expired - using position", cv::Point(20, 80), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        } else {
            cv::putText(allPolylinesViz_, "Position-based detection", cv::Point(20, 80), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
        }
    }

    return (isLeftLane);
}