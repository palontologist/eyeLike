#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>

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
  cv::namedWindow(main_window_name,CV_WINDOW_NORMAL);
  cv::moveWindow(main_window_name, 400, 100);
  cv::namedWindow(face_window_name,CV_WINDOW_NORMAL);
  cv::moveWindow(face_window_name, 10, 100);
  cv::namedWindow("Right Eye",CV_WINDOW_NORMAL);
  cv::moveWindow("Right Eye", 10, 600);
  cv::namedWindow("Left Eye",CV_WINDOW_NORMAL);
  cv::moveWindow("Left Eye", 10, 800);

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
            }
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
      }
      else {
        printf(" --(!) No captured frame -- Break!");
        break;
      }

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
        // Print health summary to console
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

  imshow(face_window_name, faceROI);
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
