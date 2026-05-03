#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <cmath>
#include <cstdio>
#include <sstream>
#include <iomanip>

#include "exercises.h"
#include "constants.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static cv::Scalar colWhite(255, 255, 255);
static cv::Scalar colBlack(0, 0, 0);
static cv::Scalar colGreen(0, 200, 0);
static cv::Scalar colRed(0, 0, 220);
static cv::Scalar colYellow(0, 220, 220);
static cv::Scalar colCyan(220, 220, 0);

// Draw a semi-transparent filled rectangle.
static void drawTransparentRect(cv::Mat &img, cv::Rect r,
                                cv::Scalar color, double alpha) {
    cv::Mat overlay;
    img.copyTo(overlay);
    cv::rectangle(overlay, r, color, -1);
    cv::addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, img);
}

// Draw centred text with a shadow.
static void drawCenteredText(cv::Mat &img, const std::string &text,
                             cv::Point center, double scale,
                             cv::Scalar color, int thickness = 1) {
    int baseline = 0;
    cv::Size sz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale,
                                  thickness, &baseline);
    cv::Point org(center.x - sz.width / 2, center.y + sz.height / 2);
    // shadow
    cv::putText(img, text, org + cv::Point(1, 1),
                cv::FONT_HERSHEY_SIMPLEX, scale, colBlack, thickness + 1);
    cv::putText(img, text, org,
                cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness);
}

// ---------------------------------------------------------------------------
// Saccade target layout (normalised 0-1 within the face rect)
// ---------------------------------------------------------------------------
static const double kSaccNormX[5] = { 0.50, 0.15, 0.85, 0.50, 0.50 };
static const double kSaccNormY[5] = { 0.40, 0.40, 0.40, 0.20, 0.60 };

// ---------------------------------------------------------------------------
// ExerciseEngine implementation
// ---------------------------------------------------------------------------

ExerciseEngine::ExerciseEngine()
    : state(STATE_IDLE),
      currentType(EXERCISE_SACCADE),
      stateStartTime(0.0), sessionStartTime(0.0),
      framesTotal(0), framesOnTarget(0),
      calibSamples(0),
      saccadeStep(0),
      focusPhase(0), phaseStartTime(0.0),
      reminderTriggerTime(0.0), reminderPending(false) {
    lastResult.type          = EXERCISE_SACCADE;
    lastResult.name          = "";
    lastResult.score         = 0.0;
    lastResult.framesTotal   = 0;
    lastResult.framesOnTarget = 0;
}

// ---------------------------------------------------------------------------

void ExerciseEngine::startExercise(ExerciseType type, double t) {
    currentType      = type;
    state            = STATE_CALIBRATE;
    stateStartTime   = t;
    framesTotal      = 0;
    framesOnTarget   = 0;
    calibCenter      = cv::Point2f(0, 0);
    calibSamples     = 0;
    saccadeStep      = 0;
    focusPhase       = 0;
    phaseStartTime   = t;
    if (sessionStartTime == 0.0) sessionStartTime = t;
    // Schedule first 20-20-20 reminder
    if (reminderTriggerTime == 0.0)
        reminderTriggerTime = t + kReminder202020IntervalSec;
}

void ExerciseEngine::nextExercise(double t) {
    int next = (static_cast<int>(currentType) + 1) % static_cast<int>(EXERCISE_COUNT);
    startExercise(static_cast<ExerciseType>(next), t);
}

void ExerciseEngine::abort() {
    state = STATE_IDLE;
}

bool ExerciseEngine::isActive()       const { return state != STATE_IDLE; }
ExerciseState ExerciseEngine::getState()    const { return state; }
ExerciseType  ExerciseEngine::getCurrentType() const { return currentType; }
ExerciseResult ExerciseEngine::getLastResult() const { return lastResult; }
double ExerciseEngine::getSessionStart() const { return sessionStartTime; }

// ---------------------------------------------------------------------------

cv::Point ExerciseEngine::toFrame(cv::Point faceRel, cv::Rect faceRect) const {
    return cv::Point(faceRect.x + faceRel.x, faceRect.y + faceRel.y);
}

cv::Point ExerciseEngine::currentTargetFaceRel(cv::Rect faceRect,
                                                double t) const {
    switch (currentType) {
    case EXERCISE_SACCADE: {
        int step = std::min(saccadeStep, kSaccadeCount - 1);
        return cv::Point(
            static_cast<int>(kSaccNormX[step] * faceRect.width),
            static_cast<int>(kSaccNormY[step] * faceRect.height));
    }
    case EXERCISE_SMOOTH_PURSUIT: {
        // Figure-8 (Lissajous): x = A*sin(t), y = B*sin(2t)
        double elapsed = t - stateStartTime;
        double speed   = CV_PI * 2.0 / 8.0;          // full cycle every 8 s
        double rx      = faceRect.width  * 0.30;
        double ry      = faceRect.height * 0.15;
        int cx         = static_cast<int>(faceRect.width  * 0.50);
        int cy         = static_cast<int>(faceRect.height * 0.40);
        return cv::Point(
            cx + static_cast<int>(rx * std::sin(speed * elapsed)),
            cy + static_cast<int>(ry * std::sin(2.0 * speed * elapsed)));
    }
    default:
        // Focus shift, palming, and reminder don't use a specific gaze target.
        return cv::Point(static_cast<int>(faceRect.width  * 0.50),
                         static_cast<int>(faceRect.height * 0.40));
    }
}

// ---------------------------------------------------------------------------

void ExerciseEngine::update(cv::Point leftPupil, cv::Point rightPupil,
                             cv::Rect faceRect, cv::Size /*frameSize*/,
                             double t) {
    if (sessionStartTime == 0.0) sessionStartTime = t;

    // Automatic 20-20-20 reminder check (independent of exercise state)
    if (reminderTriggerTime > 0.0 && t >= reminderTriggerTime &&
        !reminderPending && state == STATE_IDLE) {
        reminderPending    = true;
        currentType        = EXERCISE_REMINDER_20_20;
        state              = STATE_EXERCISE;
        stateStartTime     = t;
        framesTotal        = 0;
        framesOnTarget     = 0;
    }

    if (state == STATE_IDLE) return;

    double elapsed = t - stateStartTime;

    // -----------------------------------------------------------------------
    // State transitions
    // -----------------------------------------------------------------------
    if (state == STATE_CALIBRATE) {
        // Accumulate calibration data
        double avgX = (leftPupil.x + rightPupil.x) * 0.5;
        calibCenter.x = (calibCenter.x * calibSamples + static_cast<float>(avgX)) /
                        static_cast<float>(calibSamples + 1);
        calibCenter.y = (calibCenter.y * calibSamples + static_cast<float>(leftPupil.y)) /
                        static_cast<float>(calibSamples + 1);
        ++calibSamples;

        if (elapsed >= kCalibrateDurationSec) {
            state          = STATE_EXERCISE;
            stateStartTime = t;
            phaseStartTime = t;
        }
        return;
    }

    if (state == STATE_REST) {
        if (elapsed >= kExerciseRestDurationSec) {
            state = STATE_RESULTS;
        }
        return;
    }

    if (state == STATE_RESULTS) {
        // Stay in results for 4 seconds, then return to IDLE
        if (elapsed >= 4.0) {
            state = STATE_IDLE;
            if (reminderPending) {
                reminderPending     = false;
                reminderTriggerTime = t + kReminder202020IntervalSec;
            }
        }
        return;
    }

    // -----------------------------------------------------------------------
    // STATE_EXERCISE — per-exercise logic
    // -----------------------------------------------------------------------
    double exerciseDur = (currentType == EXERCISE_PALMING ||
                          currentType == EXERCISE_REMINDER_20_20)
                         ? kPalmingDurationSec
                         : kExerciseDurationSec;

    if (elapsed >= exerciseDur) {
        finishExercise(t);
        return;
    }

    ++framesTotal;

    // Update saccade step counter
    if (currentType == EXERCISE_SACCADE) {
        int step = static_cast<int>(elapsed / kSaccadeStepDurationSec);
        saccadeStep = step % kSaccadeCount;
    }

    // Update focus-shift phase
    if (currentType == EXERCISE_FOCUS_SHIFT) {
        double phaseElapsed = t - phaseStartTime;
        if (phaseElapsed >= kFocusShiftPhaseDurationSec) {
            focusPhase     = 1 - focusPhase;
            phaseStartTime = t;
        }
    }

    // Scoring for gaze-tracking exercises (saccade & smooth pursuit)
    if (currentType == EXERCISE_SACCADE ||
        currentType == EXERCISE_SMOOTH_PURSUIT) {
        cv::Point target = currentTargetFaceRel(faceRect, t);
        // Average pupil position in face-relative coordinates
        cv::Point avgPupil((leftPupil.x + rightPupil.x) / 2,
                           (leftPupil.y + rightPupil.y) / 2);
        double dist      = cv::norm(avgPupil - target);
        double tolerance = faceRect.width * kExerciseToleranceFraction;
        if (dist < tolerance) ++framesOnTarget;
    }
}

// ---------------------------------------------------------------------------

void ExerciseEngine::finishExercise(double t) {
    ExerciseResult r;
    r.type          = currentType;
    r.name          = exerciseName(currentType);
    r.framesTotal   = framesTotal;
    r.framesOnTarget = framesOnTarget;
    r.score         = (framesTotal > 0)
                      ? 100.0 * framesOnTarget / framesTotal
                      : 100.0;  // non-gaze exercises default to 100 %

    if (r.score >= 75.0)
        r.feedback = "Great job! Your gaze tracking is excellent.";
    else if (r.score >= 50.0)
        r.feedback = "Good effort. Keep practising to improve accuracy.";
    else
        r.feedback = "Keep going — consistent practice builds eye strength.";

    lastResult     = r;
    state          = STATE_REST;
    stateStartTime = t;
}

// ---------------------------------------------------------------------------

std::string ExerciseEngine::exerciseName(ExerciseType t) {
    switch (t) {
    case EXERCISE_SACCADE:        return "Saccadic Exercise";
    case EXERCISE_SMOOTH_PURSUIT: return "Smooth Pursuit";
    case EXERCISE_FOCUS_SHIFT:    return "Focus Shifting";
    case EXERCISE_PALMING:        return "Palming Rest";
    case EXERCISE_REMINDER_20_20: return "20-20-20 Reminder";
    default:                      return "Exercise";
    }
}

std::string ExerciseEngine::exerciseInstruction(ExerciseType t) {
    switch (t) {
    case EXERCISE_SACCADE:
        return "Move your eyes to the GREEN dot as quickly as possible";
    case EXERCISE_SMOOTH_PURSUIT:
        return "Smoothly follow the GREEN dot with your eyes";
    case EXERCISE_FOCUS_SHIFT:
        return "Alternate focus: look at the screen, then a distant object";
    case EXERCISE_PALMING:
        return "Close your eyes, cup your palms over them, and relax";
    case EXERCISE_REMINDER_20_20:
        return "Look at something 20 feet away for 20 seconds";
    default:
        return "";
    }
}

// ---------------------------------------------------------------------------
// drawOverlay
// ---------------------------------------------------------------------------

void ExerciseEngine::drawOverlay(cv::Mat &frame, cv::Rect faceRect,
                                 double t) const {
    if (state == STATE_IDLE) return;

    int W = frame.cols;
    int H = frame.rows;
    double elapsed = t - stateStartTime;

    // ---- Top banner (exercise name + state) ----
    drawTransparentRect(frame, cv::Rect(0, 0, W, 50), colBlack, 0.55);
    std::string banner = exerciseName(currentType);
    switch (state) {
    case STATE_CALIBRATE: banner += "  [CALIBRATE]"; break;
    case STATE_EXERCISE:  banner += "  [ACTIVE]";    break;
    case STATE_REST:      banner += "  [REST]";      break;
    case STATE_RESULTS:   banner += "  [RESULTS]";   break;
    default: break;
    }
    drawCenteredText(frame, banner, cv::Point(W / 2, 26), 0.7, colYellow, 1);

    // ---- Instruction bar ----
    if (state == STATE_CALIBRATE || state == STATE_EXERCISE) {
        std::string instr = (state == STATE_CALIBRATE)
            ? "Look at the centre dot and hold still..."
            : exerciseInstruction(currentType);
        drawTransparentRect(frame, cv::Rect(0, H - 45, W, 45), colBlack, 0.55);
        drawCenteredText(frame, instr, cv::Point(W / 2, H - 20), 0.55, colWhite);
    }

    // ---- Progress bar ----
    if (state == STATE_EXERCISE) {
        double dur = (currentType == EXERCISE_PALMING ||
                      currentType == EXERCISE_REMINDER_20_20)
                     ? kPalmingDurationSec : kExerciseDurationSec;
        double pct = std::min(elapsed / dur, 1.0);
        int barW   = static_cast<int>(pct * W);
        cv::rectangle(frame, cv::Rect(0, 50, barW, 6), colGreen, -1);
        cv::rectangle(frame, cv::Rect(0, 50, W, 6),
                      cv::Scalar(80, 80, 80), 1);

        // Countdown timer
        int remaining = static_cast<int>(std::ceil(dur - elapsed));
        std::ostringstream oss;
        oss << remaining << "s";
        cv::putText(frame, oss.str(), cv::Point(W - 50, 48),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, colYellow, 1);
    }

    // ---- Calibration dot ----
    if (state == STATE_CALIBRATE && faceRect.width > 0) {
        cv::Point centre = toFrame(
            cv::Point(static_cast<int>(faceRect.width  * 0.5),
                      static_cast<int>(faceRect.height * 0.4)),
            faceRect);
        cv::circle(frame, centre, 12, colGreen, -1);
        cv::circle(frame, centre, 14, colWhite, 2);
        // Countdown ring
        int r = 30;
        double frac = std::min(elapsed / kCalibrateDurationSec, 1.0);
        cv::ellipse(frame, centre, cv::Size(r, r), -90.0,
                    0.0, frac * 360.0, colYellow, 2);
    }

    // ---- Exercise-specific overlay ----
    if (state == STATE_EXERCISE) {
        switch (currentType) {

        case EXERCISE_SACCADE:
        case EXERCISE_SMOOTH_PURSUIT: {
            // Draw all saccade targets dimly, then the current one brightly
            if (currentType == EXERCISE_SACCADE && faceRect.width > 0) {
                for (int i = 0; i < kSaccadeCount; ++i) {
                    cv::Point p = toFrame(
                        cv::Point(
                            static_cast<int>(kSaccNormX[i] * faceRect.width),
                            static_cast<int>(kSaccNormY[i] * faceRect.height)),
                        faceRect);
                    cv::Scalar c = (i == saccadeStep % kSaccadeCount)
                                   ? colGreen
                                   : cv::Scalar(60, 60, 60);
                    int rad = (i == saccadeStep % kSaccadeCount) ? 14 : 7;
                    cv::circle(frame, p, rad, c, -1);
                    if (i == saccadeStep % kSaccadeCount)
                        cv::circle(frame, p, rad + 2, colWhite, 2);
                }
            } else if (currentType == EXERCISE_SMOOTH_PURSUIT &&
                       faceRect.width > 0) {
                cv::Point tgt = toFrame(
                    currentTargetFaceRel(faceRect, t), faceRect);
                cv::circle(frame, tgt, 14, colGreen, -1);
                cv::circle(frame, tgt, 16, colWhite, 2);
                // Trailing ghost
                double ghostT = t - 0.15;
                if (ghostT > stateStartTime) {
                    cv::Point ghost = toFrame(
                        currentTargetFaceRel(faceRect, ghostT), faceRect);
                    cv::circle(frame, ghost, 7, cv::Scalar(0, 100, 0), -1);
                }
            }
            break;
        }

        case EXERCISE_FOCUS_SHIFT: {
            // Alternate "NEAR" / "FAR" label with a big indicator
            std::string label = (focusPhase == 0) ? "NEAR (look at screen)"
                                                  : "FAR  (look away)";
            cv::Scalar  clr   = (focusPhase == 0) ? colCyan : colYellow;
            drawTransparentRect(frame,
                cv::Rect(W / 4, H / 3, W / 2, H / 5), colBlack, 0.60);
            drawCenteredText(frame, label,
                cv::Point(W / 2, H / 3 + H / 10), 0.8, clr, 1);
            // Phase countdown
            double phaseElapsed = t - phaseStartTime;
            int rem = static_cast<int>(
                std::ceil(kFocusShiftPhaseDurationSec - phaseElapsed));
            std::ostringstream oss;
            oss << rem << "s";
            drawCenteredText(frame, oss.str(),
                cv::Point(W / 2, H / 3 + H / 10 + 36), 0.7, colWhite);
            break;
        }

        case EXERCISE_PALMING:
        case EXERCISE_REMINDER_20_20: {
            int remaining = static_cast<int>(
                std::ceil(kPalmingDurationSec - elapsed));
            std::ostringstream oss;
            oss << remaining;
            drawTransparentRect(frame,
                cv::Rect(W / 2 - 70, H / 2 - 80, 140, 120), colBlack, 0.65);
            drawCenteredText(frame, oss.str(),
                cv::Point(W / 2, H / 2 - 20), 3.0, colYellow, 2);
            if (currentType == EXERCISE_REMINDER_20_20)
                drawCenteredText(frame, "Look 20 ft away",
                    cv::Point(W / 2, H / 2 + 50), 0.65, colCyan);
            else
                drawCenteredText(frame, "Eyes closed + palms",
                    cv::Point(W / 2, H / 2 + 50), 0.65, colCyan);
            break;
        }

        default: break;
        }
    }

    // ---- REST screen ----
    if (state == STATE_REST) {
        int remaining = static_cast<int>(
            std::ceil(kExerciseRestDurationSec - elapsed));
        drawTransparentRect(frame, cv::Rect(W / 4, H / 4, W / 2, H / 2),
                            colBlack, 0.65);
        drawCenteredText(frame, "REST", cv::Point(W / 2, H / 2 - 30),
                         1.2, colGreen, 2);
        std::ostringstream oss;
        oss << remaining << "s";
        drawCenteredText(frame, oss.str(), cv::Point(W / 2, H / 2 + 20),
                         0.8, colWhite);
    }

    // ---- RESULTS screen ----
    if (state == STATE_RESULTS) {
        drawTransparentRect(frame, cv::Rect(W / 6, H / 5, W * 2 / 3, H * 3 / 5),
                            colBlack, 0.75);
        drawCenteredText(frame, lastResult.name,
                         cv::Point(W / 2, H / 5 + 36), 0.8, colYellow, 1);

        std::ostringstream scoreStr;
        scoreStr << std::fixed << std::setprecision(1)
                 << lastResult.score << "%";
        cv::Scalar scoreClr = (lastResult.score >= 75.0) ? colGreen
                            : (lastResult.score >= 50.0) ? colYellow
                            : colRed;
        drawCenteredText(frame, scoreStr.str(),
                         cv::Point(W / 2, H / 2 - 10), 2.0, scoreClr, 2);
        drawCenteredText(frame, lastResult.feedback,
                         cv::Point(W / 2, H / 2 + 48), 0.55, colWhite);
        drawCenteredText(frame, "Press [E] for next exercise",
                         cv::Point(W / 2, H * 4 / 5 - 10), 0.5,
                         cv::Scalar(180, 180, 180));
    }
}
