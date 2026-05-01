#ifndef EXERCISES_H
#define EXERCISES_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <string>

enum ExerciseType {
    EXERCISE_SACCADE        = 0,  // Jump gaze to fixed target dots
    EXERCISE_SMOOTH_PURSUIT = 1,  // Track a moving target
    EXERCISE_FOCUS_SHIFT    = 2,  // Alternate near/far focus
    EXERCISE_PALMING        = 3,  // Eye rest with palms
    EXERCISE_REMINDER_20_20 = 4,  // 20-20-20 look-away reminder
    EXERCISE_COUNT          = 5
};

enum ExerciseState {
    STATE_IDLE      = 0,
    STATE_CALIBRATE,  // Record baseline pupil position
    STATE_EXERCISE,   // Active exercise
    STATE_REST,       // Brief rest between exercises
    STATE_RESULTS     // Show score overlay
};

// Result for a completed exercise.
struct ExerciseResult {
    ExerciseType type;
    std::string  name;
    double       score;          // 0–100 %
    int          framesTotal;
    int          framesOnTarget;
    std::string  feedback;
};

class ExerciseEngine {
public:
    ExerciseEngine();

    // Start the specified exercise (resets state machine).
    void startExercise(ExerciseType type, double timestampSec);

    // Advance to the next exercise type (wraps around).
    void nextExercise(double timestampSec);

    // Update per-frame.
    //   leftPupil / rightPupil : face-relative pixel coordinates
    //   faceRect               : face bounding box in the full frame
    //   frameSize              : full-frame dimensions
    void update(cv::Point leftPupil, cv::Point rightPupil,
                cv::Rect faceRect, cv::Size frameSize, double timestampSec);

    // Draw exercise overlay onto a BGR frame.
    void drawOverlay(cv::Mat &frame, cv::Rect faceRect, double timestampSec) const;

    bool          isActive()         const;
    ExerciseState getState()         const;
    ExerciseType  getCurrentType()   const;
    ExerciseResult getLastResult()   const;
    double        getSessionStart()  const;

    // Cancel the current exercise and return to IDLE.
    void abort();

private:
    ExerciseState  state;
    ExerciseType   currentType;
    double         stateStartTime;
    double         sessionStartTime;

    // Scoring counters
    int framesTotal;
    int framesOnTarget;

    // Calibration baseline (face-relative, average of left+right X and left Y)
    cv::Point2f calibCenter;
    int         calibSamples;

    // Saccade: index into target array
    int saccadeStep;
    // Number of saccade targets
    static const int kSaccadeCount = 5;

    // Smooth-pursuit: phase offset driven by elapsed time
    // Focus-shift: phase (0 = near, 1 = far)
    int    focusPhase;
    double phaseStartTime;

    // 20-20-20 automatic reminder
    double reminderTriggerTime;
    bool   reminderPending;

    ExerciseResult lastResult;

    // Helpers
    // Returns the current target in face-relative pixel coords.
    cv::Point currentTargetFaceRel(cv::Rect faceRect, double timestampSec) const;
    // Converts a face-relative point to full-frame coords.
    cv::Point toFrame(cv::Point faceRel, cv::Rect faceRect) const;
    void finishExercise(double timestampSec);
    static std::string exerciseName(ExerciseType t);
    static std::string exerciseInstruction(ExerciseType t);
};

#endif
