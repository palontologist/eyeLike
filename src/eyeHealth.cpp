#include "opencv2/imgproc/imgproc.hpp"
#include <cmath>
#include <algorithm>
#include <sstream>

#include "eyeHealth.h"
#include "constants.h"

EyeHealthMonitor::EyeHealthMonitor()
    : inBlink(false), blinkStartTime(0.0),
      totalBlinks(0),
      sessionStartTime(0.0), lastTimestamp(0.0) {}

void EyeHealthMonitor::update(cv::Point leftPupil, cv::Point rightPupil,
                               int eyeROIHeightL, int eyeROIHeightR,
                               double timestampSec) {
    if (sessionStartTime == 0.0) sessionStartTime = timestampSec;
    lastTimestamp = timestampSec;

    // --- Blink detection: short Y-history ---
    leftYHistory.push_back(static_cast<double>(leftPupil.y));
    rightYHistory.push_back(static_cast<double>(rightPupil.y));
    if (leftYHistory.size()  > static_cast<size_t>(kBlinkWindowFrames)) leftYHistory.pop_front();
    if (rightYHistory.size() > static_cast<size_t>(kBlinkWindowFrames)) rightYHistory.pop_front();

    // --- Gaze stability: longer history ---
    leftXStab.push_back(static_cast<double>(leftPupil.x));
    leftYStab.push_back(static_cast<double>(leftPupil.y));
    rightXStab.push_back(static_cast<double>(rightPupil.x));
    rightYStab.push_back(static_cast<double>(rightPupil.y));
    if (leftXStab.size()  > static_cast<size_t>(kGazeStabilityWindowFrames)) leftXStab.pop_front();
    if (leftYStab.size()  > static_cast<size_t>(kGazeStabilityWindowFrames)) leftYStab.pop_front();
    if (rightXStab.size() > static_cast<size_t>(kGazeStabilityWindowFrames)) rightXStab.pop_front();
    if (rightYStab.size() > static_cast<size_t>(kGazeStabilityWindowFrames)) rightYStab.pop_front();

    // --- Eye openness proxy (pupil Y / ROI height) ---
    if (eyeROIHeightL > 0)
        leftOpennessHist.push_back(static_cast<double>(leftPupil.y) / eyeROIHeightL);
    if (eyeROIHeightR > 0)
        rightOpennessHist.push_back(static_cast<double>(rightPupil.y) / eyeROIHeightR);
    if (leftOpennessHist.size()  > static_cast<size_t>(kGazeStabilityWindowFrames)) leftOpennessHist.pop_front();
    if (rightOpennessHist.size() > static_cast<size_t>(kGazeStabilityWindowFrames)) rightOpennessHist.pop_front();

    detectBlink(timestampSec);
}

void EyeHealthMonitor::detectBlink(double timestampSec) {
    if (leftYHistory.size() < static_cast<size_t>(kBlinkWindowFrames)) return;

    double stdDev = std::sqrt(computeVariance(leftYHistory));

    if (!inBlink && stdDev > kBlinkYStdDevThreshold) {
        inBlink        = true;
        blinkStartTime = timestampSec;
    } else if (inBlink && stdDev <= kBlinkYStdDevThreshold) {
        inBlink = false;
        double durationMs = (timestampSec - blinkStartTime) * 1000.0;
        // Sanity window: 50–1000 ms
        if (durationMs >= 50.0 && durationMs <= 1000.0) {
            ++totalBlinks;
            blinkDurations.push_back(durationMs);
        }
    }
}

double EyeHealthMonitor::computeMean(const std::deque<double> &v) const {
    if (v.empty()) return 0.0;
    double sum = 0.0;
    for (double x : v) sum += x;
    return sum / static_cast<double>(v.size());
}

double EyeHealthMonitor::computeVariance(const std::deque<double> &v) const {
    if (v.size() < 2) return 0.0;
    double mean = computeMean(v);
    double var  = 0.0;
    for (double x : v) var += (x - mean) * (x - mean);
    return var / static_cast<double>(v.size());
}

BlinkStats EyeHealthMonitor::getBlinkStats() const {
    BlinkStats s;
    s.totalBlinks = totalBlinks;

    double elapsedMin = (lastTimestamp - sessionStartTime) / 60.0;
    s.blinksPerMinute = (elapsedMin > 0.0) ? totalBlinks / elapsedMin : 0.0;

    if (!blinkDurations.empty()) {
        double sum = 0.0;
        for (double d : blinkDurations) sum += d;
        s.avgBlinkDurationMs = sum / static_cast<double>(blinkDurations.size());
    } else {
        s.avgBlinkDurationMs = 0.0;
    }

    // Only flag after at least 30 s of data
    s.lowBlinkRate  = (elapsedMin > 0.5) && (s.blinksPerMinute < 10.0);
    s.highBlinkRate = (s.blinksPerMinute > 30.0);
    return s;
}

PupilSymmetry EyeHealthMonitor::getPupilSymmetry() const {
    PupilSymmetry sym;
    sym.leftOpennessRatio  = leftOpennessHist.empty()  ? 0.5 : computeMean(leftOpennessHist);
    sym.rightOpennessRatio = rightOpennessHist.empty() ? 0.5 : computeMean(rightOpennessHist);
    double diff = std::abs(sym.leftOpennessRatio - sym.rightOpennessRatio);
    sym.asymmetryDetected  = (diff > kPupilAsymmetryThreshold);
    return sym;
}

GazeStability EyeHealthMonitor::getGazeStability() const {
    GazeStability gs;
    gs.varianceX = (computeVariance(leftXStab) + computeVariance(rightXStab)) * 0.5;
    gs.varianceY = (computeVariance(leftYStab) + computeVariance(rightYStab)) * 0.5;
    gs.nystagmusDetected =
        (gs.varianceX > kNystagmusVarianceThreshold ||
         gs.varianceY > kNystagmusVarianceThreshold);
    return gs;
}

HealthSummary EyeHealthMonitor::getSummary() const {
    HealthSummary summary;
    summary.blinks    = getBlinkStats();
    summary.symmetry  = getPupilSymmetry();
    summary.stability = getGazeStability();

    if (summary.blinks.lowBlinkRate)
        summary.advice.push_back(
            "Blink rate is below normal (<10/min). Consider lubricating eye drops.");
    if (summary.blinks.highBlinkRate)
        summary.advice.push_back(
            "Blink rate is high (>30/min). Possible eye irritation or fatigue.");
    if (summary.blinks.avgBlinkDurationMs > 400.0)
        summary.advice.push_back(
            "Average blink duration is prolonged (>400 ms). This may indicate ptosis.");
    if (summary.symmetry.asymmetryDetected)
        summary.advice.push_back(
            "Eye openness asymmetry detected. Consider consulting an eye specialist.");
    if (summary.stability.nystagmusDetected)
        summary.advice.push_back(
            "Irregular gaze oscillations detected. Rest your eyes and try again.");
    if (summary.advice.empty())
        summary.advice.push_back(
            "Eye metrics look normal. Keep up the good work!");

    return summary;
}

void EyeHealthMonitor::reset(double timestampSec) {
    leftYHistory.clear();
    rightYHistory.clear();
    leftXStab.clear();
    leftYStab.clear();
    rightXStab.clear();
    rightYStab.clear();
    leftOpennessHist.clear();
    rightOpennessHist.clear();
    inBlink        = false;
    totalBlinks    = 0;
    blinkDurations.clear();
    sessionStartTime = timestampSec;
    lastTimestamp    = timestampSec;
}
