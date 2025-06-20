#include "KalmanFilter.hpp"

KalmanFilter::KalmanFilter() : leftLaneInitialized(false), rightLaneInitialized(false), 
                    midLaneInitialized(false) {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::init() {
    // For each lane, track 4 polynomial coefficients and their derivatives
    // State: [a, b, c, d, da/dt, db/dt, dc/dt, dd/dt] for x = ay³ + by² + cy + d
    const int stateSize = 8;
    const int measSize = 4;  // We measure polynomial coefficients
    const int contrSize = 0; // No control input
    
    // Initialize both Kalman filters
    leftLaneKF = cv::KalmanFilter(stateSize, measSize, contrSize, CV_32F);
    rightLaneKF = cv::KalmanFilter(stateSize, measSize, contrSize, CV_32F);
    middleLaneKF = cv::KalmanFilter(stateSize, measSize, contrSize, CV_32F);
    
    // Set transition matrix (constant velocity model for coefficients)
    cv::setIdentity(leftLaneKF.transitionMatrix);
    cv::setIdentity(rightLaneKF.transitionMatrix);
    cv::setIdentity(middleLaneKF.transitionMatrix);
    
    // Add velocity components
    for (int i = 0; i < 4; i++) {
        leftLaneKF.transitionMatrix.at<float>(i, i+4) = 1.0;
        rightLaneKF.transitionMatrix.at<float>(i, i+4) = 1.0;
        middleLaneKF.transitionMatrix.at<float>(i, i+4) = 1.0;
    }
    
    // Set measurement matrix (we only measure coefficients, not their derivatives)
    leftLaneKF.measurementMatrix = cv::Mat::zeros(measSize, stateSize, CV_32F);
    rightLaneKF.measurementMatrix = cv::Mat::zeros(measSize, stateSize, CV_32F);
    middleLaneKF.measurementMatrix = cv::Mat::zeros(measSize, stateSize, CV_32F);
    
    for (int i = 0; i < measSize; i++) {
        leftLaneKF.measurementMatrix.at<float>(i, i) = 1.0f;
        rightLaneKF.measurementMatrix.at<float>(i, i) = 1.0f;
        middleLaneKF.measurementMatrix.at<float>(i, i) = 1.0f;
    }
    
    // Lower process noise for polynomials (they change more slowly)
    cv::setIdentity(leftLaneKF.processNoiseCov, cv::Scalar(1e-3));
    cv::setIdentity(rightLaneKF.processNoiseCov, cv::Scalar(1e-3));
    cv::setIdentity(middleLaneKF.processNoiseCov, cv::Scalar(1e-3));
    
    // Measurement noise
    cv::setIdentity(leftLaneKF.measurementNoiseCov, cv::Scalar(0.01));
    cv::setIdentity(rightLaneKF.measurementNoiseCov, cv::Scalar(0.01));
    cv::setIdentity(middleLaneKF.measurementNoiseCov, cv::Scalar(0.01));
    
    // Initial state covariance
    cv::setIdentity(leftLaneKF.errorCovPost, cv::Scalar(1));
    cv::setIdentity(rightLaneKF.errorCovPost, cv::Scalar(1));
    cv::setIdentity(middleLaneKF.errorCovPost, cv::Scalar(1));
}

cv::Mat KalmanFilter::polyfit(const cv::Mat& y_vals, const cv::Mat& x_vals, int degree)
{
    // Check if we have enough points for the requested degree
    if (y_vals.rows < degree + 1)
    {
        // Not enough points - reduce degree or return empty matrix
        if (y_vals.rows < 2)
        {
            return cv::Mat();
        }
        // Adjust degree based on available points
        degree = y_vals.rows - 1;
    }

    // Ensure data is in the right format (CV_64F)
    cv::Mat y_vals_64f, x_vals_64f;
    y_vals.convertTo(y_vals_64f, CV_64F);
    x_vals.convertTo(x_vals_64f, CV_64F);

    // Create the design matrix with appropriate dimensions
    cv::Mat A = cv::Mat::zeros(y_vals_64f.rows, degree + 1, CV_64F);

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
    cv::Mat coeffs;
    try
    {
        solve(A, x_vals_64f, coeffs, cv::DECOMP_SVD);
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "Error in polyfit: " << e.what() << std::endl;
        return cv::Mat();
    }
    if (abs(coeffs.at<double>(0)) < 0.0005)
        coeffs.at<double>(0) = 0;
    if (abs(coeffs.at<double>(1)) < 0.001)
        coeffs.at<double>(1) = 0;
    if (abs(coeffs.at<double>(2)) < 0.001)
        coeffs.at<double>(2) = 0;

    return coeffs;
}


cv::Mat KalmanFilter::extractPolynomialCoefficients(const std::vector<cv::Point>& laneCurve) {
    if (laneCurve.size() < 4) return cv::Mat();
    
    // Prepare data for polyfit
    cv::Mat xVals(laneCurve.size(), 1, CV_64F);
    cv::Mat yVals(laneCurve.size(), 1, CV_64F);
    
    for (size_t i = 0; i < laneCurve.size(); i++) {
        xVals.at<double>(i) = laneCurve[i].x;
        yVals.at<double>(i) = laneCurve[i].y;
    }
    
    // Fit polynomial (y = ax² + bx + c)
    // Note: We're fitting x as a function of y since lanes are more vertical
    return polyfit(yVals, xVals, 3);
}

std::vector<cv::Point> KalmanFilter::reconstructLaneFromCoefficients(const cv::Mat& coeffs, int height, int width) {
    std::vector<cv::Point> curve;
    
    // Generate points along the polynomial curve
    for (int y = 0; y < height; y += 10) {
        double a = coeffs.at<float>(0);
        double b = coeffs.at<float>(1);
        double c = coeffs.at<float>(2);
        double d = coeffs.at<float>(3);
        
        // x = ay² + by + c
        int x = static_cast<int>(a*y*y*y + b*y*y + c*y + d);
        
        // Ensure point is within frame
        if (x >= 0 && x < width) {
            curve.push_back(cv::Point(x, y));
        }
    }
    
    return curve;
}

void KalmanFilter::updateLeftLaneFilter(const std::vector<cv::Point>& lane) {
    // Extract polynomial coefficients
    leftLaneInitialized = true;
    cv::Mat coeffs = extractPolynomialCoefficients(lane);
    
    if (coeffs.empty()) return;
    
    // Convert to measurement format
    cv::Mat measurement(4, 1, CV_32F);
    for (int i = 0; i < 4; i++) {
        measurement.at<float>(i) = static_cast<float>(coeffs.at<double>(i));
    }
    leftLaneCoeffs = coeffs.clone();
    // Update Kalman filter
    leftLaneKF.correct(measurement);
}


void KalmanFilter::updateRightLaneFilter(const std::vector<cv::Point>& lane) {
    
    rightLaneInitialized = true;
    // Extract polynomial coefficients
    cv::Mat coeffs = extractPolynomialCoefficients(lane);
    
    if (coeffs.empty()) return;
    
    // Convert to measurement format
    cv::Mat measurement(4, 1, CV_32F);
    for (int i = 0; i < 4; i++) {
        measurement.at<float>(i) = static_cast<float>(coeffs.at<double>(i));
    }
    rightLaneCoeffs = coeffs.clone();
    // Update Kalman filter
    rightLaneKF.correct(measurement);
}

void KalmanFilter::updateMiddleLaneFilter(const std::vector<cv::Point>& lane) {
    
    midLaneInitialized = true;
    // Extract polynomial coefficients
    cv::Mat coeffs = extractPolynomialCoefficients(lane);
    
    if (coeffs.empty()) return;
    
    // Convert to measurement format
    cv::Mat measurement(4, 1, CV_32F);
    for (int i = 0; i < 4; i++) {
        measurement.at<float>(i) = static_cast<float>(coeffs.at<double>(i));
    }
    middleLaneCoeffs = coeffs.clone();
    // Update Kalman filter
    middleLaneKF.correct(measurement);
}

std::vector<cv::Point> KalmanFilter::predictLeftLaneCurve(int height, int width) {
    cv::Mat leftPrediction = leftLaneKF.predict();
    return (reconstructLaneFromCoefficients(leftPrediction, height, width));
}

std::vector<cv::Point> KalmanFilter::predictRightLaneCurve(int height, int width) {
    cv::Mat rightPrediction = rightLaneKF.predict();
    return (reconstructLaneFromCoefficients(rightPrediction, height, width));
}

std::vector<cv::Point> KalmanFilter::predictMiddleLaneCurve(int height, int width) {
    cv::Mat middlePrediction = middleLaneKF.predict();
    return (reconstructLaneFromCoefficients(middlePrediction, height, width));
}