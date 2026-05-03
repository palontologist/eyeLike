#ifndef CONSTANTS_H
#define CONSTANTS_H

// Debugging
const bool kPlotVectorField = false;

// Size constants
const int kEyePercentTop = 25;
const int kEyePercentSide = 13;
const int kEyePercentHeight = 30;
const int kEyePercentWidth = 35;

// Preprocessing
const bool kSmoothFaceImage = false;
const float kSmoothFaceFactor = 0.005;

// Algorithm Parameters
const int kFastEyeWidth = 50;
const int kWeightBlurSize = 5;
const bool kEnableWeight = true;
const float kWeightDivisor = 1.0;
const double kGradientThreshold = 50.0;

// Postprocessing
const bool kEnablePostProcess = true;
const float kPostProcessThreshold = 0.97;

// Eye Corner
const bool kEnableEyeCorner = false;

// ---- Eye Health Monitoring ----
// Blink detection: rolling window length (frames) and Y-stddev trigger
const int    kBlinkWindowFrames       = 8;
const double kBlinkYStdDevThreshold   = 3.0;
// Gaze-stability window (frames) and nystagmus variance trigger (px²)
const int    kGazeStabilityWindowFrames    = 30;
const double kNystagmusVarianceThreshold   = 25.0;
// Eye-openness asymmetry threshold (fraction of ROI height, 0-1)
const double kPupilAsymmetryThreshold      = 0.15;

// ---- Exercise Engine ----
const double kCalibrateDurationSec         = 3.0;   // look-at-center warm-up
const double kExerciseDurationSec          = 30.0;  // seconds per exercise set
const double kExerciseRestDurationSec      = 5.0;   // rest between exercises
const double kExerciseToleranceFraction    = 0.04;  // fraction of face width: on-target radius
const double kSaccadeStepDurationSec       = 2.0;   // seconds per saccade target
const double kFocusShiftPhaseDurationSec   = 10.0;  // near/far phase length
const double kPalmingDurationSec           = 20.0;  // palming rest countdown
const double kReminder202020IntervalSec    = 1200.0;// 20 min between reminders
const double kReminder202020DurationSec    = 20.0;  // 20-second look-away timer

#endif