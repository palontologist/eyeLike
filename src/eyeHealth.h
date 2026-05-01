#ifndef EYE_HEALTH_H
#define EYE_HEALTH_H

#include <opencv2/core/core.hpp>
#include <deque>
#include <vector>
#include <string>

// Aggregated blink statistics for the current session.
struct BlinkStats {
    int    totalBlinks;
    double blinksPerMinute;
    double avgBlinkDurationMs;
    bool   lowBlinkRate;   // < 10/min — possible screen-induced dry eye
    bool   highBlinkRate;  // > 30/min — possible eye irritation or fatigue
};

// Compares average eye openness (pupil Y fraction within eye ROI) between eyes.
struct PupilSymmetry {
    double leftOpennessRatio;   // 0-1, fraction of eye-ROI height
    double rightOpennessRatio;
    bool   asymmetryDetected;   // difference exceeds kPupilAsymmetryThreshold
};

// Rolling gaze-position variance used to flag nystagmus-like oscillations.
struct GazeStability {
    double varianceX;           // px² average of left+right X histories
    double varianceY;           // px² average of left+right Y histories
    bool   nystagmusDetected;   // exceeds kNystagmusVarianceThreshold
};

// Full per-session summary with plain-language advice strings.
struct HealthSummary {
    BlinkStats     blinks;
    PupilSymmetry  symmetry;
    GazeStability  stability;
    std::vector<std::string> advice;
};

class EyeHealthMonitor {
public:
    EyeHealthMonitor();

    // Call once per frame with face-relative pupil coordinates.
    // eyeROIHeightL/R: pixel height of the respective eye ROI.
    void update(cv::Point leftPupil, cv::Point rightPupil,
                int eyeROIHeightL, int eyeROIHeightR, double timestampSec);

    BlinkStats    getBlinkStats()     const;
    PupilSymmetry getPupilSymmetry()  const;
    GazeStability getGazeStability()  const;
    HealthSummary getSummary()        const;

    // Reset all accumulators (start a fresh session).
    void reset(double timestampSec);

private:
    // Blink detection
    std::deque<double> leftYHistory;
    std::deque<double> rightYHistory;
    bool   inBlink;
    double blinkStartTime;
    int    totalBlinks;
    std::vector<double> blinkDurations;

    // Gaze stability (longer rolling window)
    std::deque<double> leftXStab;
    std::deque<double> leftYStab;
    std::deque<double> rightXStab;
    std::deque<double> rightYStab;

    // Eye openness proxy (pupil-Y / ROI-height)
    std::deque<double> leftOpennessHist;
    std::deque<double> rightOpennessHist;

    double sessionStartTime;
    double lastTimestamp;

    void   detectBlink(double timestampSec);
    double computeMean(const std::deque<double> &v) const;
    double computeVariance(const std::deque<double> &v) const;
};

#endif
