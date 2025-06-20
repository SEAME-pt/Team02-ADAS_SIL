#include "LaneDetector.hpp"

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
    outputData = new float[2 * HEIGHT * WIDTH];

    std::memset(inputData, 0, 3 * HEIGHT * WIDTH * sizeof(float));
    std::memset(outputData, 0, 2 * HEIGHT * WIDTH * sizeof(float));
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
    if (abs(coeffs.at<double>(0)) < 0.01)
        coeffs.at<double>(0) = 0;

    return coeffs;
}


cv::Mat LaneDetector::preProcess(const cv::Mat& frame)
{
    static cv::Mat resized(HEIGHT, WIDTH, CV_8UC3);
    static cv::Mat float_mat(HEIGHT, WIDTH, CV_32FC3);
    
    // Use INTER_NEAREST for faster resizing
    cv::resize(frame, resized, cv::Size(WIDTH, HEIGHT), 0, 0,
    cv::INTER_NEAREST);
    cv::Mat rgb_image;
    cv::cvtColor(resized, rgb_image, cv::COLOR_BGR2RGB);
    return rgb_image;
}

void LaneDetector::postProcess(cv::Mat& frame)
{
    static cv::Mat mask(HEIGHT, WIDTH, CV_8UC1);
    // static cv::Mat resized_mask;
    // static cv::Mat colored_mask;

    uchar* mask_data       = mask.data;
    const int total_pixels = HEIGHT * WIDTH;

    for (int i = 0; i < total_pixels; i++)
    {
        float p0     = outputData[i];
        float p1     = outputData[total_pixels + i];
        mask_data[i] = (p0 > p1) ? 255 : 0;
    }

    // Apply ROI directly to the mask before collecting points
    int height = mask.rows;
    int width = mask.cols;

    // Define trapezoidal ROI (same as in filterPoints)
    float bottomY = height;
    float topY = height * 0.40f;
    
    float bottomLeftX = width * - 1.5f;
    float bottomRightX = width * 2.5f;

    bottomLeftX = max(0, int(width * -1.5));
    bottomRightX = min(width-1, int(width * 2.5));
    
    float topLeftX = width * 0.3f;
    float topRightX = width * 0.7f;

    // Create a copy of the mask for ROI application
    cv::Mat roiMask = mask.clone();

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
    lanePoints.reserve(maskPoints.size()); // Pre-allocate for performance
    float x_scale = static_cast<float>(frame.cols) / mask.cols;
    float y_scale = static_cast<float>(frame.rows) / mask.rows;
    // std::cout << "Scaling factors: x=" << x_scale << ", y=" << y_scale << std::endl;

    for (const auto& pt : maskPoints)
    {
        int scaledX = pt.x * x_scale;
        int scaledY = pt.y * y_scale;
        lanePoints.push_back(cv::Point(scaledX, scaledY));
    }
    createLanes(lanePoints, frame);
}

void LaneDetector::createLanes(std::vector<cv::Point> lanePoints,
                               cv::Mat& frame)
{
    if (firstFrame)
    {
        laneWidthEstimate = frame.cols * 0.25;
        firstFrame        = false;
    }

    // Define Left and Right lanes
    // 1. Cluster points using Mean Shift
    std::vector<cv::Point> leftPoints, rightPoints;
    clusterLanePoints(lanePoints, leftPoints, rightPoints, frame);

        // SAVE LANE POINTS FOR EXTERNAL ACCESS
    this->allLanePoints = lanePoints;  // Store in class member

    // CREATE VISUALIZATION FRAME TO SHOW ALL LANE POINTS
    cv::Mat pointsVisualization = frame.clone();
    int pointCount = lanePoints.size();

    // Save the visualization for later access
    cv::resize(pointsVisualization, this->lanePointsVisualization, frame.size(), 0, 0, cv::INTER_NEAREST);

    // 2. Fit polynomial curves to each set of points
    std::vector<cv::Point> leftCurve  = fitCurveToPoints(leftPoints, frame);
    std::vector<cv::Point> rightCurve = fitCurveToPoints(rightPoints, frame);

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
                    double c = leftCoeffs.at<double>(2) + laneWidthEstimate;  // shift right
                    for (int y = height; y >= height / 3; y -= 5)
                    {
                        double x = a * y * y + b * y + c;
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
        int targetY = height - (2 * height / 3); // 2/3 up from bottom

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

    // Display lateral error as text
    std::string errorText = "Error: " + std::to_string(lateralError);
    cv::putText(frame, errorText, cv::Point(20, 90), cv::FONT_HERSHEY_SIMPLEX,
                0.7, cv::Scalar(255, 255, 255), 2);
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

int LaneDetector::cluster2DPoints(const std::vector<cv::Point>& points, 
                               std::vector<std::vector<cv::Point>>& clusters,
                               float distanceThreshold) {
    if (points.empty()) return 0;
    
    clusters.clear();
    
    // Initialize visit flags
    std::vector<bool> visited(points.size(), false);
    
    // For each unvisited point
    for (size_t i = 0; i < points.size(); i++) {
        if (visited[i]) continue;
        
        // Start a new cluster
        std::vector<cv::Point> cluster;
        std::queue<size_t> queue;
        
        // Add current point to queue and mark as visited
        queue.push(i);
        visited[i] = true;
        
        // Process all connected points
        while (!queue.empty()) {
            size_t current = queue.front();
            queue.pop();
            
            // Add to cluster
            cluster.push_back(points[current]);
            
            // Check all other points for connectivity
            for (size_t j = 0; j < points.size(); j++) {
                if (!visited[j]) {
                    // Calculate Euclidean distance
                    float dx = points[current].x - points[j].x;
                    float dy = points[current].y - points[j].y;
                    float distance = std::sqrt(dx*dx + dy*dy);
                    
                    if (distance < distanceThreshold) {
                        queue.push(j);
                        visited[j] = true;
                    }
                }
            }
        }
        
        // Add cluster if it has enough points (more than 5)
        if (cluster.size() >= 5) {
            clusters.push_back(cluster);
        }
    }
    
    // Sort clusters by size (largest first)
    std::sort(clusters.begin(), clusters.end(), 
              [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                  return a.size() > b.size();
              });
    
    return clusters.size();
}

// Main function: Cluster lane points into left and right using the expected boundaries.
void LaneDetector::clusterLanePoints(const std::vector<cv::Point>& points,
                                     std::vector<cv::Point>& leftPoints,
                                     std::vector<cv::Point>& rightPoints,
                                     cv::Mat& frame)
{
    // Clear outputs
    leftPoints.clear();
    rightPoints.clear();
    int width = frame.cols;
    int height = frame.rows;
    int midX = width / 2;

    std::vector<cv::Point> filtered = points;
    // Stage 1: Filter points (ROI + density)
    // std::vector<cv::Point> filtered = filterPoints(points, frame);
    if (filtered.empty()) return; // no valid points

    // *** NEW STAGE 1B: DETECT SEPARATION USING 2D CLUSTERING***
    bool naturalSeparationFound = false;
    int separationPoint = midX;

    // If we have enough points, try 2D clustering
    if (filtered.size() >= 10) {
        // Cluster points based on 2D proximity
        std::vector<std::vector<cv::Point>> clusters;
        int clusterCount = cluster2DPoints(filtered, clusters, width * 0.005f);
        // std::cout << "CLUSTER COUNT == " << clusterCount << std::endl;
        // If we found exactly 2 clusters, they're likely left and right lanes
        if (clusterCount == 2 && !clusters[0].empty() && !clusters[1].empty()) {
            // Calculate centroid of each cluster
            cv::Point centroid1(0, 0), centroid2(0, 0);
            for (const auto& pt : clusters[0]) {
                centroid1.x += pt.x;
                centroid1.y += pt.y;
            }
            centroid1.x /= clusters[0].size();
            centroid1.y /= clusters[0].size();
            
            for (const auto& pt : clusters[1]) {
                centroid2.x += pt.x;
                centroid2.y += pt.y;
            }
            centroid2.x /= clusters[1].size();
            centroid2.y /= clusters[1].size();
            
            // Determine midpoint between clusters
            separationPoint = (centroid1.x + centroid2.x) / 2;
            naturalSeparationFound = true;
            
            // Draw the detected separation
            cv::line(frame, cv::Point(separationPoint, height), 
                    cv::Point(separationPoint, height/2),
                    cv::Scalar(255, 0, 255), 2);
            
            // Add text to show the gap size
            int gapSize = std::abs(centroid1.x - centroid2.x);
            std::string gapText = "Gap: " + std::to_string(gapSize);
            cv::putText(frame, gapText, cv::Point(separationPoint - 50, height/2 + 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 255), 1);

             // DIRECTLY ASSIGN POINTS BASED ON CLUSTERS
            // Determine which cluster is left vs right
            if (centroid1.x < centroid2.x) {
                leftPoints = clusters[0];
                rightPoints = clusters[1];
            } else {
                leftPoints = clusters[1];
                rightPoints = clusters[0];
            }
            
            // Skip other assignment methods since we already have our clusters
            for (const auto& pt : leftPoints) {
                cv::circle(frame, pt, 4, cv::Scalar(255, 100, 100), -1); // Light red for left points
            }

            for (const auto& pt : rightPoints) {
                cv::circle(frame, pt, 4, cv::Scalar(100, 255, 100), -1); // Light green for right points
            }
            
            // Update lane width estimate based on the clustering
            laneWidthEstimate = 0.3 * gapSize + 0.7 * laneWidthEstimate;
            
            // Add debug info
            std::string methodText = "Using 2D Clusters";
            cv::putText(frame, methodText, cv::Point(20, 120), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
                        
            // Since we've assigned points directly, skip to Stage 6 to update history
        }
        // If clustering didn't give useful results, fall back to simple gap detection
        else {
            // Original gap detection code:
            std::vector<int> xCoordinates;
            for (const auto& pt : filtered) {
                xCoordinates.push_back(pt.x);
            }
            std::sort(xCoordinates.begin(), xCoordinates.end());
            
            int maxGap = 0;
            int gapPosition = -1;
            int minGapWidth = width * 0.05f; // Minimum meaningful gap (5% of width)
            
            for (size_t i = 1; i < xCoordinates.size(); i++) {
                int gap = xCoordinates[i] - xCoordinates[i-1];
                if (gap > maxGap && gap > minGapWidth) {
                    int midGapPosition = (xCoordinates[i] + xCoordinates[i-1]) / 2;
                    if (midGapPosition > width * 0.25 && midGapPosition < width * 0.75) {
                        maxGap = gap;
                        gapPosition = midGapPosition;
                    }
                }
            }
            
            if (maxGap > 0 && gapPosition > 0) {
                naturalSeparationFound = true;
                separationPoint = gapPosition;
                
                cv::line(frame, cv::Point(separationPoint, height), 
                        cv::Point(separationPoint, height/2),
                        cv::Scalar(255, 0, 255), 2);
                
                std::string gapText = "Gap: " + std::to_string(maxGap);
                cv::putText(frame, gapText, cv::Point(separationPoint - 50, height/2 + 30), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 255), 1);
            }

            if (laneWidthEstimate <= 0) {
                laneWidthEstimate = width * 0.6f;
            }
            int expectedLeftBoundary, expectedRightBoundary;
            computeExpectedBoundaries(frame, midX, laneWidthEstimate,
                                      prevLeftCurve, prevRightCurve,
                                      expectedLeftBoundary, expectedRightBoundary);
            // Set tolerance to about 20% of lane width.
            float tolerance = laneWidthEstimate * 0.2f;
        
            // (Optional) Visual debug: Draw expected boundaries
            cv::line(frame, cv::Point(expectedLeftBoundary, height), cv::Point(expectedLeftBoundary, height/2),
                     cv::Scalar(128, 0, 255), 2);
            cv::line(frame, cv::Point(expectedRightBoundary, height), cv::Point(expectedRightBoundary, height/2),
                     cv::Scalar(0, 128, 255), 2);
        
            // Stage 3: Assign points to lanes based on expected boundaries
            assignPointsToLanes(filtered, leftPoints, rightPoints, expectedLeftBoundary, expectedRightBoundary, tolerance, midX);
        
            // Stage 4: Sanity Check – verify that the left cluster is actually on the left.
            if (!leftPoints.empty() && !rightPoints.empty()) {
                double leftMean = 0, rightMean = 0;
                for (auto pt : leftPoints) leftMean += pt.x;
                for (auto pt : rightPoints) rightMean += pt.x;
                leftMean /= leftPoints.size();
                rightMean /= rightPoints.size();
                if (leftMean > rightMean)
                    std::swap(leftPoints, rightPoints);
            }
        
            // Stage 5: Fallback to history if clusters are too sparse.
            if (leftPoints.size() < 3 && !prevLeftPoints.empty())
                leftPoints = prevLeftPoints;
            if (rightPoints.size() < 3 && !prevRightPoints.empty())
                rightPoints = prevRightPoints;
        }
    }

    for (const auto& pt : leftPoints) {
        cv::circle(frame, pt, 4, cv::Scalar(255, 100, 100), -1); // Light red for left points
    }

    for (const auto& pt : rightPoints) {
        cv::circle(frame, pt, 4, cv::Scalar(100, 255, 100), -1); // Light green for right points
    }

    // Stage 2: Compute expected boundaries from history and laneWidthEstimate
    // Ensure laneWidthEstimate has a default (if not yet set).
    
    // Stage 6: Update history for next frame.
    if (leftPoints.size() >= 3)
        prevLeftPoints = leftPoints;
    if (rightPoints.size() >= 3)
        prevRightPoints = rightPoints;

    std::string methodText = naturalSeparationFound ? "Separation Detected" : "Using History/Default";
    cv::putText(frame, methodText, cv::Point(20, 120), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
}


std::vector<cv::Point>
LaneDetector::fitCurveToPoints(const std::vector<cv::Point>& points,
                               cv::Mat& frame)
{
    if (points.size() < 3)
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
    int degree = 2; // Quadratic curve
    cv::Mat coeffs = polyfit(y_mat, x_mat, degree);

    // Generate smooth curve
    std::vector<cv::Point> curve;
    if (!coeffs.empty() && coeffs.rows >= 3)
    {
        for (int y = frame.rows; y >= frame.rows * 0.35; y -= 5)
        {
            double x = coeffs.at<double>(0) * y * y + coeffs.at<double>(1) * y +
                       coeffs.at<double>(2);

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
        this->leftCoeffs = polyfit(y_mat, x_mat, 2);
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
        rightCoeffs = polyfit(y_mat, x_mat, 2);
    }
}
