#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <thread>
#include <atomic>

// HEADLESS MODE: No GUI windows, saves frames to disk
#define HEADLESS_MODE 0

#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"
#include "eyeHealth.h"
#include "exercises.h"
#include "sessionLogger.h"

/* Attempt at supporting openCV version 4.0.1 or higher */
#if CV_MAJOR_VERSION >= 4
#define CV_WINDOW_NORMAL                cv::WINDOW_NORMAL
#define CV_BGR2YCrCb                    cv::COLOR_BGR2YCrCb
#define CV_HAAR_SCALE_IMAGE             cv::CASCADE_SCALE_IMAGE
#define CV_HAAR_FIND_BIGGEST_OBJECT     cv::CASCADE_FIND_BIGGEST_OBJECT
#endif


/** Constants **/


/** Per-frame eye data returned by findEyes / detectAndDisplay */
struct EyeData {
    cv::Point leftPupil;    // face-relative pixel coords
    cv::Point rightPupil;
    cv::Rect  faceRect;     // face bounding box in the full frame
    int       eyeROIHeightL;
    int       eyeROIHeightR;
    bool      faceDetected;
};

/** Function Headers */
EyeData detectAndDisplay( cv::Mat frame );

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
cv::String face_cascade_name = "../res/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;
std::string main_window_name = "Capture - Face detection";
std::string face_window_name = "Capture - Face";
  cv::RNG rng(12345);
  cv::Mat debugImage;
  cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);
  
  // Frame counter for headless mode
  int frameCounter = 0;
  int savedFrameCount = 0;
  
  // Show summary overlay flag
  bool showSummaryOverlay = false;
  double summaryDisplayStartTime = 0.0;

// Health & exercise singletons
EyeHealthMonitor healthMonitor;
ExerciseEngine   exerciseEngine;
SessionLogger    sessionLogger;
std::vector<ExerciseResult> sessionResults;
double hudTimerStart = 0.0;  // wall-clock start for HUD elapsed timer

/**
 * @function main
 */
int main( int argc, const char** argv ) {
  cv::Mat frame;

  // Load the cascades
  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n"); return -1; };
  #if !HEADLESS_MODE
  cv::namedWindow(main_window_name,CV_WINDOW_NORMAL);
  cv::moveWindow(main_window_name, 400, 100);
  cv::resizeWindow(main_window_name, 1280, 720);
  #endif

  createCornerKernels();
  ellipse(skinCrCbHist, cv::Point(113, 155), cv::Size(23, 15),
          43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);

  printf("Controls: [e] start/next exercise  [h] print health summary  [q] quit & save log\n");

  // I make an attempt at supporting both 2.x and 3.x OpenCV
#if CV_MAJOR_VERSION < 3
  CvCapture* capture = cvCaptureFromCAM( 0 );
  if( capture ) {
    while( true ) {
      frame = cvQueryFrame( capture );
#else
  cv::VideoCapture capture(0);
  if( capture.isOpened() ) {
    while( true ) {
      capture.read(frame);
#endif
      // mirror it
      cv::flip(frame, frame, 1);
      frame.copyTo(debugImage);

      // Apply the classifier to the frame
      if( !frame.empty() ) {
        double timestamp = static_cast<double>(cv::getTickCount()) /
                           cv::getTickFrequency();
        if (hudTimerStart == 0.0) hudTimerStart = timestamp;

        EyeData eyeData = detectAndDisplay( frame );

        if (eyeData.faceDetected) {
          healthMonitor.update(eyeData.leftPupil, eyeData.rightPupil,
                               eyeData.eyeROIHeightL, eyeData.eyeROIHeightR,
                               timestamp);

          ExerciseState prevState = exerciseEngine.getState();
          exerciseEngine.update(eyeData.leftPupil, eyeData.rightPupil,
                                eyeData.faceRect, frame.size(), timestamp);

          // Draw exercise overlay on top of debugImage
          exerciseEngine.drawOverlay(debugImage, eyeData.faceRect, timestamp);

          // Collect completed exercise result (only on the transition into STATE_RESULTS)
          ExerciseState es = exerciseEngine.getState();
          if (prevState != STATE_RESULTS && es == STATE_RESULTS) {
            ExerciseResult r = exerciseEngine.getLastResult();
            if (r.framesTotal > 0) {
              sessionResults.push_back(r);
              sessionLogger.logExerciseResult(r);
              // Auto-show summary when exercise completes
              showSummaryOverlay = true;
              summaryDisplayStartTime = timestamp;
            }
          }
          
          // Auto-hide summary after 8 seconds
          if (showSummaryOverlay && (timestamp - summaryDisplayStartTime) > 8.0) {
            showSummaryOverlay = false;
          }

          sessionLogger.logFrame(timestamp,
                                 eyeData.leftPupil.x,  eyeData.leftPupil.y,
                                 eyeData.rightPupil.x, eyeData.rightPupil.y,
                                 healthMonitor.isInBlink());
        }

        // HUD: blink count overlay
        {
          BlinkStats    bs = healthMonitor.getBlinkStats();
          GazeStability gs = healthMonitor.getGazeStability();
          PupilSymmetry ps = healthMonitor.getPupilSymmetry();

          // Line 1: blink count
          char buf[128];
          snprintf(buf, sizeof(buf), "Blinks: %d  (%.1f/min)",
                   bs.totalBlinks, bs.blinksPerMinute);
          cv::putText(debugImage, buf, cv::Point(8, debugImage.rows - 30),
                      cv::FONT_HERSHEY_SIMPLEX, 0.5,
                      cv::Scalar(0, 220, 220), 1);

          // Line 2: stability | symmetry | elapsed time
          double elapsedSec = static_cast<double>(cv::getTickCount()) /
                              cv::getTickFrequency() - hudTimerStart;
          if (elapsedSec < 0.0) elapsedSec = 0.0;
          int eMin = static_cast<int>(elapsedSec) / 60;
          int eSec = static_cast<int>(elapsedSec) % 60;

          cv::Scalar stabClr = gs.nystagmusDetected
                               ? cv::Scalar(0, 0, 220)
                               : cv::Scalar(0, 220, 0);
          cv::Scalar symClr  = ps.asymmetryDetected
                               ? cv::Scalar(0, 0, 220)
                               : cv::Scalar(0, 220, 0);

          char stabBuf[32], symBuf[32], timeBuf[32];
          snprintf(stabBuf, sizeof(stabBuf), "Stability: %s",
                   gs.nystagmusDetected ? "UNSTABLE" : "OK");
          snprintf(symBuf, sizeof(symBuf), "Sym: %s",
                   ps.asymmetryDetected ? "ASYM" : "OK");
          snprintf(timeBuf, sizeof(timeBuf), "%02d:%02d", eMin, eSec);

          int y2 = debugImage.rows - 8;
          cv::putText(debugImage, stabBuf, cv::Point(8, y2),
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, stabClr, 1);
          cv::putText(debugImage, symBuf, cv::Point(200, y2),
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, symClr, 1);
          cv::putText(debugImage, timeBuf, cv::Point(debugImage.cols - 70, y2),
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 220, 220), 1);
          
          // Eye width measurements (if face detected)
          if (eyeData.faceDetected) {
            int leftEyeWidth = eyeData.eyeROIHeightL;
            int rightEyeWidth = eyeData.eyeROIHeightR;
            char eyeBuf[128];
            snprintf(eyeBuf, sizeof(eyeBuf), "Eyes: L=%dpx R=%dpx",
                     leftEyeWidth, rightEyeWidth);
            cv::putText(debugImage, eyeBuf, cv::Point(debugImage.cols - 220, 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(220, 220, 0), 1);
          }
        }

        // Show summary overlay (after exercise completion or 'h' key)
        if (showSummaryOverlay) {
          HealthSummary summary = healthMonitor.getSummary();
          BlinkStats    &bs     = summary.blinks;
          PupilSymmetry &ps     = summary.symmetry;
          GazeStability &gs     = summary.stability;
          
          // Semi-transparent background
          cv::Mat overlay;
          debugImage.copyTo(overlay);
          cv::rectangle(overlay, cv::Rect(80, 80, debugImage.cols - 160, debugImage.rows - 160),
                        cv::Scalar(0, 0, 0), -1);
          cv::addWeighted(overlay, 0.85, debugImage, 0.15, 0, debugImage);
          
          int y = 120;
          cv::Scalar white(255, 255, 255);
          cv::Scalar yellow(0, 220, 220);
          cv::Scalar green(0, 220, 0);
          cv::Scalar red(0, 0, 220);
          
          // Title
          cv::putText(debugImage, "EYE HEALTH SUMMARY", 
                      cv::Point(debugImage.cols / 2 - 180, y),
                      cv::FONT_HERSHEY_SIMPLEX, 0.9, yellow, 2);
          y += 50;
          
          // Blink stats
          char buf[256];
          snprintf(buf, sizeof(buf), "Blinks: %d (%.1f/min, avg %.0f ms)",
                   bs.totalBlinks, bs.blinksPerMinute, bs.avgBlinkDurationMs);
          cv::putText(debugImage, buf, cv::Point(120, y),
                      cv::FONT_HERSHEY_SIMPLEX, 0.6, white, 1);
          y += 35;
          
          snprintf(buf, sizeof(buf), "Low blink rate:  %s", bs.lowBlinkRate ? "YES" : "no");
          cv::Scalar blinkClr = bs.lowBlinkRate ? red : green;
          cv::putText(debugImage, buf, cv::Point(120, y), cv::FONT_HERSHEY_SIMPLEX, 0.55, blinkClr, 1);
          y += 30;
          
          snprintf(buf, sizeof(buf), "High blink rate: %s", bs.highBlinkRate ? "YES" : "no");
          blinkClr = bs.highBlinkRate ? red : green;
          cv::putText(debugImage, buf, cv::Point(120, y), cv::FONT_HERSHEY_SIMPLEX, 0.55, blinkClr, 1);
          y += 40;
          
          // Pupil symmetry
          snprintf(buf, sizeof(buf), "Pupil openness: L=%.2f  R=%.2f", 
                   ps.leftOpennessRatio, ps.rightOpennessRatio);
          cv::putText(debugImage, buf, cv::Point(120, y),
                      cv::FONT_HERSHEY_SIMPLEX, 0.6, white, 1);
          y += 35;
          
          snprintf(buf, sizeof(buf), "Asymmetry: %s", ps.asymmetryDetected ? "DETECTED" : "Normal");
          cv::Scalar symClr = ps.asymmetryDetected ? red : green;
          cv::putText(debugImage, buf, cv::Point(120, y), cv::FONT_HERSHEY_SIMPLEX, 0.55, symClr, 1);
          y += 40;
          
          // Gaze stability
          snprintf(buf, sizeof(buf), "Gaze variance: X=%.1f  Y=%.1f", gs.varianceX, gs.varianceY);
          cv::putText(debugImage, buf, cv::Point(120, y),
                      cv::FONT_HERSHEY_SIMPLEX, 0.6, white, 1);
          y += 35;
          
          snprintf(buf, sizeof(buf), "Nystagmus: %s", gs.nystagmusDetected ? "DETECTED" : "Normal");
          cv::Scalar stabClr = gs.nystagmusDetected ? red : green;
          cv::putText(debugImage, buf, cv::Point(120, y), cv::FONT_HERSHEY_SIMPLEX, 0.55, stabClr, 1);
          y += 50;
          
          // Advice
          cv::putText(debugImage, "Advice:", cv::Point(120, y),
                      cv::FONT_HERSHEY_SIMPLEX, 0.6, yellow, 1);
          y += 30;
          for (const std::string &a : summary.advice) {
            cv::putText(debugImage, "- " + a, cv::Point(140, y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, white, 1);
            y += 25;
          }
          
          // Close instruction
          cv::putText(debugImage, "Press [H] to close or wait 8 seconds",
                      cv::Point(debugImage.cols / 2 - 200, debugImage.rows - 120),
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(180, 180, 180), 1);
        }

        if (!eyeData.faceDetected) {
          // "No face detected" centred warning
          const std::string noFaceMsg = "No face detected";
          int baseline = 0;
          cv::Size sz = cv::getTextSize(noFaceMsg, cv::FONT_HERSHEY_SIMPLEX,
                                        0.8, 2, &baseline);
          cv::Point org((debugImage.cols - sz.width) / 2,
                        (debugImage.rows + sz.height) / 2);
          cv::putText(debugImage, noFaceMsg, org + cv::Point(1, 1),
                      cv::FONT_HERSHEY_SIMPLEX, 0.8,
                      cv::Scalar(0, 0, 0), 3);
          cv::putText(debugImage, noFaceMsg, org,
                      cv::FONT_HERSHEY_SIMPLEX, 0.8,
                      cv::Scalar(0, 80, 255), 2);
        }

        // Show startup instructions overlay for first 10 seconds
        {
          double elapsedSec = timestamp - hudTimerStart;
          if (elapsedSec < 10.0) {
            // Semi-transparent overlay
            cv::Mat overlay;
            debugImage.copyTo(overlay);
            cv::rectangle(overlay, cv::Rect(50, 50, debugImage.cols - 100, debugImage.rows - 100),
                          cv::Scalar(0, 0, 0), -1);
            cv::addWeighted(overlay, 0.7, debugImage, 0.3, 0, debugImage);
            
            // Title
            cv::putText(debugImage, "EyeLike - Eye Health & Exercise App",
                        cv::Point(debugImage.cols / 2 - 280, 100),
                        cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 220, 220), 2);
            
            // Instructions
            std::vector<std::string> instructions = {
              "This app tracks your eyes and guides you through exercises",
              "",
              "CONTROLS:",
              "  [E]  Start exercise / Next exercise",
              "  [H]  Print health summary",
              "  [Q]  Quit and save session log",
              "",
              "EXERCISES:",
              "  1. Saccadic - Look at green dots as they appear",
              "  2. Smooth Pursuit - Follow the moving green dot",
              "  3. Focus Shift - Alternate near/far focus",
              "  4. Palming - Rest eyes with palms over them",
              "  5. 20-20-20 - Look 20 feet away for 20 seconds",
              "",
              "Press [E] to start your first exercise!"
            };
            
            int y = 150;
            for (const auto& line : instructions) {
              cv::Scalar color = cv::Scalar(255, 255, 255);
              if (line.find("[E]") != std::string::npos || line.find("Press [E]") != std::string::npos) {
                color = cv::Scalar(0, 255, 0);
              }
              cv::putText(debugImage, line, cv::Point(80, y),
                          cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 1);
              y += 28;
            }
          }
        }
      }
      else {
        printf(" --(!) No captured frame -- Break!");
        break;
      }

      #if HEADLESS_MODE
      // Save frames periodically in headless mode
      frameCounter++;
      if (frameCounter % 30 == 0) {
        char filename[64];
        snprintf(filename, sizeof(filename), "frame_%04d.png", savedFrameCount++);
        imwrite(filename, debugImage);
        printf("[Headless] Saved %s\n", filename);
      }
      
      // Auto-exercise mode: start saccade exercise at frame 100
      if (frameCounter == 100) {
        double timestamp = static_cast<double>(cv::getTickCount()) /
                           cv::getTickFrequency();
        exerciseEngine.startExercise(EXERCISE_SACCADE, timestamp);
        printf("[Exercise] Auto-starting: Saccadic Exercise\n");
      }
      
      // Auto-quit after 500 frames (~50 seconds)
      if (frameCounter >= 500) {
        printf("[Headless] Auto-quit after 500 frames\n");
        HealthSummary summary = healthMonitor.getSummary();
        sessionLogger.writeReport(summary, sessionResults);
        printf("[Session] Log saved to: %s\n",
               sessionLogger.getFilePath().c_str());
        break;
      }
      
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      #else
      imshow(main_window_name,debugImage);

      int c = cv::waitKey(10);
      if( (char)c == 'c' ) { break; }
      if( (char)c == 'f' ) {
        imwrite("frame.png",frame);
      }
      if( (char)c == 'e' ) {
        // Start or advance to next exercise
        double timestamp = static_cast<double>(cv::getTickCount()) /
                           cv::getTickFrequency();
        if (!exerciseEngine.isActive()) {
          exerciseEngine.startExercise(EXERCISE_SACCADE, timestamp);
          printf("[Exercise] Starting: Saccadic Exercise\n");
        } else {
          exerciseEngine.nextExercise(timestamp);
          printf("[Exercise] Next exercise started.\n");
        }
      }
      if( (char)c == 'h' ) {
        // Toggle health summary overlay
        showSummaryOverlay = !showSummaryOverlay;
        if (showSummaryOverlay) {
          summaryDisplayStartTime = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();
        }
        
        // Also print to console
        HealthSummary summary = healthMonitor.getSummary();
        BlinkStats    &bs     = summary.blinks;
        PupilSymmetry &ps     = summary.symmetry;
        GazeStability &gs     = summary.stability;
        printf("\n===== Eye Health Summary =====\n");
        printf("  Blinks: %d (%.1f/min, avg %.0f ms)\n",
               bs.totalBlinks, bs.blinksPerMinute, bs.avgBlinkDurationMs);
        printf("  Low blink rate:  %s\n", bs.lowBlinkRate  ? "YES" : "no");
        printf("  High blink rate: %s\n", bs.highBlinkRate ? "YES" : "no");
        printf("  Pupil openness L/R: %.2f / %.2f  asymmetry: %s\n",
               ps.leftOpennessRatio, ps.rightOpennessRatio,
               ps.asymmetryDetected ? "YES" : "no");
        printf("  Gaze variance X=%.1f Y=%.1f  nystagmus: %s\n",
               gs.varianceX, gs.varianceY,
               gs.nystagmusDetected ? "YES" : "no");
        printf("  Advice:\n");
        for (const std::string &a : summary.advice)
          printf("    - %s\n", a.c_str());
        printf("==============================\n\n");
      }
      if( (char)c == 'q' ) {
        // Save session report and quit
        HealthSummary summary = healthMonitor.getSummary();
        sessionLogger.writeReport(summary, sessionResults);
        printf("[Session] Log saved to: %s\n",
               sessionLogger.getFilePath().c_str());
        break;
      }
      #endif
    }
  }

  releaseCornerKernels();

  return 0;
}

EyeData findEyes(cv::Mat frame_gray, cv::Rect face) {
  EyeData data;
  data.faceRect     = face;
  data.faceDetected = true;

  cv::Mat faceROI = frame_gray(face);
  cv::Mat debugFace = faceROI;

  if (kSmoothFaceImage) {
    double sigma = kSmoothFaceFactor * face.width;
    GaussianBlur( faceROI, faceROI, cv::Size( 0, 0 ), sigma);
  }
  //-- Find eye regions and draw them
  int eye_region_width = face.width * (kEyePercentWidth/100.0);
  int eye_region_height = face.width * (kEyePercentHeight/100.0);
  int eye_region_top = face.height * (kEyePercentTop/100.0);
  cv::Rect leftEyeRegion(face.width*(kEyePercentSide/100.0),
                         eye_region_top,eye_region_width,eye_region_height);
  cv::Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide/100.0),
                          eye_region_top,eye_region_width,eye_region_height);

  data.eyeROIHeightL = eye_region_height;
  data.eyeROIHeightR = eye_region_height;

  //-- Find Eye Centers
  cv::Point leftPupil = findEyeCenter(faceROI,leftEyeRegion,"Left Eye");
  cv::Point rightPupil = findEyeCenter(faceROI,rightEyeRegion,"Right Eye");
  // get corner regions
  cv::Rect leftRightCornerRegion(leftEyeRegion);
  leftRightCornerRegion.width -= leftPupil.x;
  leftRightCornerRegion.x += leftPupil.x;
  leftRightCornerRegion.height /= 2;
  leftRightCornerRegion.y += leftRightCornerRegion.height / 2;
  cv::Rect leftLeftCornerRegion(leftEyeRegion);
  leftLeftCornerRegion.width = leftPupil.x;
  leftLeftCornerRegion.height /= 2;
  leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;
  cv::Rect rightLeftCornerRegion(rightEyeRegion);
  rightLeftCornerRegion.width = rightPupil.x;
  rightLeftCornerRegion.height /= 2;
  rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;
  cv::Rect rightRightCornerRegion(rightEyeRegion);
  rightRightCornerRegion.width -= rightPupil.x;
  rightRightCornerRegion.x += rightPupil.x;
  rightRightCornerRegion.height /= 2;
  rightRightCornerRegion.y += rightRightCornerRegion.height / 2;
  rectangle(debugFace,leftRightCornerRegion,200);
  rectangle(debugFace,leftLeftCornerRegion,200);
  rectangle(debugFace,rightLeftCornerRegion,200);
  rectangle(debugFace,rightRightCornerRegion,200);
  // change eye centers to face coordinates
  rightPupil.x += rightEyeRegion.x;
  rightPupil.y += rightEyeRegion.y;
  leftPupil.x += leftEyeRegion.x;
  leftPupil.y += leftEyeRegion.y;
  // draw eye centers
  circle(debugFace, rightPupil, 3, 1234);
  circle(debugFace, leftPupil, 3, 1234);

  data.leftPupil  = leftPupil;
  data.rightPupil = rightPupil;

  //-- Find Eye Corners
  if (kEnableEyeCorner) {
    cv::Point2f leftRightCorner = findEyeCorner(faceROI(leftRightCornerRegion), true, false);
    leftRightCorner.x += leftRightCornerRegion.x;
    leftRightCorner.y += leftRightCornerRegion.y;
    cv::Point2f leftLeftCorner = findEyeCorner(faceROI(leftLeftCornerRegion), true, true);
    leftLeftCorner.x += leftLeftCornerRegion.x;
    leftLeftCorner.y += leftLeftCornerRegion.y;
    cv::Point2f rightLeftCorner = findEyeCorner(faceROI(rightLeftCornerRegion), false, true);
    rightLeftCorner.x += rightLeftCornerRegion.x;
    rightLeftCorner.y += rightLeftCornerRegion.y;
    cv::Point2f rightRightCorner = findEyeCorner(faceROI(rightRightCornerRegion), false, false);
    rightRightCorner.x += rightRightCornerRegion.x;
    rightRightCorner.y += rightRightCornerRegion.y;
    circle(faceROI, leftRightCorner, 3, 200);
    circle(faceROI, leftLeftCorner, 3, 200);
    circle(faceROI, rightLeftCorner, 3, 200);
    circle(faceROI, rightRightCorner, 3, 200);
  }

  return data;
}


cv::Mat findSkin (cv::Mat &frame) {
  cv::Mat input;
  cv::Mat output = cv::Mat(frame.rows,frame.cols, CV_8U);

  cvtColor(frame, input, CV_BGR2YCrCb);

  for (int y = 0; y < input.rows; ++y) {
    const cv::Vec3b *Mr = input.ptr<cv::Vec3b>(y);
//    uchar *Or = output.ptr<uchar>(y);
    cv::Vec3b *Or = frame.ptr<cv::Vec3b>(y);
    for (int x = 0; x < input.cols; ++x) {
      cv::Vec3b ycrcb = Mr[x];
//      Or[x] = (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0) ? 255 : 0;
      if(skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) == 0) {
        Or[x] = cv::Vec3b(0,0,0);
      }
    }
  }
  return output;
}

/**
 * @function detectAndDisplay
 */
EyeData detectAndDisplay( cv::Mat frame ) {
  EyeData emptyData;
  emptyData.faceDetected = false;

  std::vector<cv::Rect> faces;
  //cv::Mat frame_gray;

  std::vector<cv::Mat> rgbChannels(3);
  cv::split(frame, rgbChannels);
  cv::Mat frame_gray = rgbChannels[2];

  //cvtColor( frame, frame_gray, CV_BGR2GRAY );
  //equalizeHist( frame_gray, frame_gray );
  //cv::pow(frame_gray, CV_64F, frame_gray);
  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150) );
//  findSkin(debugImage);

  for( int i = 0; i < (int)faces.size(); i++ )
  {
    rectangle(debugImage, faces[i], 1234);
  }
  //-- Show what you got
  if (faces.size() > 0) {
    return findEyes(frame_gray, faces[0]);
  }
  return emptyData;
}
