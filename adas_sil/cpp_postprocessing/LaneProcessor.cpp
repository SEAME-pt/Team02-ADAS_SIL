#include "LaneProcessor.hpp"
#include <algorithm>
#include <cmath>

LaneProcessor::LaneProcessor(int width, int height)
    : width(width), 
      height(height), 
      firstFrame(true),
      laneWidthEstimate(0.0f),
      lateralError(0.0f)
{
    prevMidPoint = cv::Point(-1, -1);
}

LaneResult LaneProcessor::process(const cv::Mat& mask)
{
    LaneResult result;
    
    // Extract points from the binary mask
    std::vector<cv::Point> maskPoints;
    for (int y = 0; y < mask.rows; y++) {
        for (int x = 0; x < mask.cols; x++) {
            if (mask.at<uchar>(y, x) > 0) {
                maskPoints.push_back(cv::Point(x, y));
            }
        }
    }
    
    // Process the lane points to get curves
    createLanes(maskPoints, result);
    
    return result;
}

void LaneProcessor::createLanes(std::vector<cv::Point>& lanePoints, LaneResult& result)
{
     // Initialize lane width estimate on first frame.
     for (const auto& pt : lanePoints) {
        cv::circle(frame, pt, 2, cv::Scalar(255, 255, 255), -1); 
        // White for all points
    }

    if (firstFrame)
    {
        laneWidthEstimate = frame.cols * 0.25;
        firstFrame        = false;
    }

    // Define Left and Right lanes
    // 1. Cluster points using Mean Shift
    std::vector<cv::Point> leftPoints, rightPoints;
    clusterLanePoints(lanePoints, leftPoints, rightPoints, frame);

    // 2. Fit polynomial curves to each set of points
    std::vector<cv::Point> leftCurve  = fitCurveToPoints(leftPoints, frame);
    std::vector<cv::Point> rightCurve = fitCurveToPoints(rightPoints, frame);

    // 3. Apply Kalman filtering for temporal smoothing and prediction
    // Initialize Kalman filter if this is the first good detection
    if (!kfInitialized && !leftCurve.empty() && !rightCurve.empty())
    {
        initKalmanFilters(leftCurve, rightCurve);
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

    // Publish error to control system if needed
    if (publisher_)
    {
        publisher_->publishCameraError(lateralError);
        //publisher_->publishLanes(leftCurve, rightCurve);
    }

    // Draw the final lane visualization
    drawLanes(frame, leftCurve, rightCurve);
    sendCoefs(leftCurve, rightCurve);

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

void LaneProcessor::clusterLanePoints(const std::vector<cv::Point>& points,
                                     std::vector<cv::Point>& leftPoints,
                                     std::vector<cv::Point>& rightPoints)
{
    // Clear outputs
    leftPoints.clear();
    rightPoints.clear();
    
    // Define ROI parameters
    float bottomY = height;
    float topY = height * 0.3f;
    float bottomLeftX = width * 0.1f;
    float bottomRightX = width * 0.9f;
    float topLeftX = width * 0.3f;
    float topRightX = width * 0.7f;
    
    // Filter points by ROI
    std::vector<cv::Point> filtered;
    for (const auto& pt : points) {
        if (pt.y >= topY) {
            float yRatio = (pt.y - topY) / (bottomY - topY);
            float leftBoundary = topLeftX + yRatio * (bottomLeftX - topLeftX);
            float rightBoundary = topRightX + yRatio * (bottomRightX - topRightX);
            
            if (pt.x >= leftBoundary && pt.x <= rightBoundary) {
                filtered.push_back(pt);
            }
        }
    }
    
    // If we have enough points, try 2D clustering
    if (filtered.size() >= 10) {
        // Try to cluster into left and right lanes
        std::vector<std::vector<cv::Point>> clusters;
        int clusterCount = cluster2DPoints(filtered, clusters, width * 0.03f);
        
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
            
            // Determine which cluster is left vs right
            if (centroid1.x < centroid2.x) {
                leftPoints = clusters[0];
                rightPoints = clusters[1];
            } else {
                leftPoints = clusters[1];
                rightPoints = clusters[0];
            }
            
            // Update lane width estimate
            int gapSize = std::abs(centroid1.x - centroid2.x);
            laneWidthEstimate = 0.3f * gapSize + 0.7f * laneWidthEstimate;
            
            return;
        }
    }
    
    // Fall back to simple division by x-coordinate
    int midX = width / 2;
    
    if (!prevLeftCurve.empty() && !prevRightCurve.empty()) {
        // Use previous curves to guide separation
        for (const auto& pt : filtered) {
            bool assignedToLeftLane = false;
            bool assignedToRightLane = false;
            
            // Find closest previous lane curve point
            float minLeftDist = width;
            float minRightDist = width;
            
            for (const auto& leftPt : prevLeftCurve) {
                if (std::abs(leftPt.y - pt.y) < height * 0.05f) {
                    float dist = std::abs(leftPt.x - pt.x);
                    if (dist < minLeftDist) {
                        minLeftDist = dist;
                    }
                }
            }
            
            for (const auto& rightPt : prevRightCurve) {
                if (std::abs(rightPt.y - pt.y) < height * 0.05f) {
                    float dist = std::abs(rightPt.x - pt.x);
                    if (dist < minRightDist) {
                        minRightDist = dist;
                    }
                }
            }
            
            // Assign to closest lane
            if (minLeftDist < minRightDist && minLeftDist < width * 0.2f) {
                leftPoints.push_back(pt);
                assignedToLeftLane = true;
            } else if (minRightDist < width * 0.2f) {
                rightPoints.push_back(pt);
                assignedToRightLane = true;
            }
            
            // If not assigned based on previous curves, use midpoint
            if (!assignedToLeftLane && !assignedToRightLane) {
                if (pt.x < midX) {
                    leftPoints.push_back(pt);
                } else {
                    rightPoints.push_back(pt);
                }
            }
        }
    } else {
        // No previous curves, use simple midpoint division
        for (const auto& pt : filtered) {
            if (pt.x < midX) {
                leftPoints.push_back(pt);
            } else {
                rightPoints.push_back(pt);
            }
        }
    }
    
    // Update point history
    if (leftPoints.size() >= 3)
        prevLeftPoints = leftPoints;
    if (rightPoints.size() >= 3)
        prevRightPoints = rightPoints;
}

int LaneProcessor::cluster2DPoints(const std::vector<cv::Point>& points, 
                                  std::vector<std::vector<cv::Point>>& clusters,
                                  float distanceThreshold)
{
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
        
        // Add cluster if it has enough points
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

std::vector<cv::Point> LaneProcessor::fitCurveToPoints(const std::vector<cv::Point>& points)
{
    if (points.size() < 3) {
        return std::vector<cv::Point>();
    }

    // Calculate weights based on y-position (lower points get higher weight)
    std::vector<float> weights(points.size(), 1.0f);
    for (size_t i = 0; i < points.size(); i++) {
        // Points lower in the image (higher y) get more weight
        weights[i] = std::sqrt(static_cast<float>(points[i].y) / height);
        weights[i] = std::max(0.5f, std::min(weights[i], 1.0f)); // Clamp between 0.5 and 1.0
    }

    // Extract point coordinates for polynomial fitting
    std::vector<float> x_vals, y_vals;
    for (size_t i = 0; i < points.size(); i++) {
        // Add points with their weights
        x_vals.push_back(points[i].x);
        y_vals.push_back(points[i].y);
    }

    // Convert to OpenCV matrix format
    cv::Mat x_mat(x_vals), y_mat(y_vals);
    
    // Fit a 2nd degree polynomial
    cv::Mat coeffs = polyfit(y_mat, x_mat, 2);

    // Generate curve points
    std::vector<cv::Point> curve;
    if (!coeffs.empty() && coeffs.rows >= 3) {
        double a = coeffs.at<double>(0);
        double b = coeffs.at<double>(1);
        double c = coeffs.at<double>(2);
        
        // Generate points from bottom of image up to 35% from bottom
        for (int y = height; y >= height * 0.35; y -= 5) {
            double x = a * y * y + b * y + c;
            if (!std::isnan(x) && !std::isinf(x)) {
                curve.push_back(cv::Point(round(x), y));
            }
        }
    }
    
    return curve;
}

void LaneProcessor::drawLanes(cv::Mat& frame, const LaneResult& result)
{
    // Draw left lane in blue
    if (!result.left_points.empty() && result.left_points.size() >= 2) {
        for (size_t i = 1; i < result.left_points.size(); i++) {
            cv::line(frame, result.left_points[i-1], result.left_points[i], cv::Scalar(255, 0, 0), 3);
        }
    }

    // Draw right lane in green
    if (!result.right_points.empty() && result.right_points.size() >= 2) {
        for (size_t i = 1; i < result.right_points.size(); i++) {
            cv::line(frame, result.right_points[i-1], result.right_points[i], cv::Scalar(0, 255, 0), 3);
        }
    }

    // Draw center lane in red if available
    if (!result.mid_points.empty() && result.mid_points.size() >= 2) {
        for (size_t i = 1; i < result.mid_points.size(); i++) {
            cv::line(frame, result.mid_points[i-1], result.mid_points[i], cv::Scalar(0, 0, 255), 2);
        }
    }

    // Draw reference point if available
    if (!result.mid_points.empty()) {
        int targetY = frame.rows - (frame.rows / 3);
        cv::Point referencePoint;
        float minDist = FLT_MAX;
        
        for (const auto& pt : result.mid_points) {
            float dist = std::abs(pt.y - targetY);
            if (dist < minDist) {
                minDist = dist;
                referencePoint = pt;
            }
        }
        
        cv::circle(frame, referencePoint, 8, cv::Scalar(255, 0, 255), -1);
    }

    // Display lateral error
    std::string errorText = "Error: " + std::to_string(result.lateral_error);
    cv::putText(frame, errorText, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 
                0.7, cv::Scalar(255, 255, 255), 2);
}

cv::Mat LaneProcessor::polyfit(const cv::Mat& y_vals, const cv::Mat& x_vals, int degree)
{
    // Check if we have enough points
    if (y_vals.rows < degree + 1) {
        if (y_vals.rows < 2) return cv::Mat();
        degree = y_vals.rows - 1;
    }

    // Convert to double precision
    cv::Mat y_vals_64f, x_vals_64f;
    y_vals.convertTo(y_vals_64f, CV_64F);
    x_vals.convertTo(x_vals_64f, CV_64F);

    // Create design matrix
    cv::Mat A = cv::Mat::zeros(y_vals_64f.rows, degree + 1, CV_64F);
    for (int i = 0; i < y_vals_64f.rows; i++) {
        for (int j = 0; j <= degree; j++) {
            A.at<double>(i, j) = pow(y_vals_64f.at<double>(i), degree - j);
        }
    }

    // Check for invalid values
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            if (cvIsNaN(A.at<double>(i, j)) || cvIsInf(A.at<double>(i, j))) {
                A.at<double>(i, j) = 0;
            }
        }
    }

    // Solve for coefficients using SVD for better stability
    cv::Mat coeffs;
    try {
        cv::solve(A, x_vals_64f, coeffs, cv::DECOMP_SVD);
    }
    catch (const cv::Exception& e) {
        std::cerr << "Error in polyfit: " << e.what() << std::endl;
        return cv::Mat();
    }

    return coeffs;
}
