#ifndef SESSION_LOGGER_H
#define SESSION_LOGGER_H

#include <fstream>
#include <string>
#include <vector>

#include "eyeHealth.h"
#include "exercises.h"

class SessionLogger {
public:
    // Opens a timestamped CSV file in the current working directory.
    SessionLogger();
    ~SessionLogger();

    // Append a per-frame data row.
    void logFrame(double timestamp,
                  int leftPupilX, int leftPupilY,
                  int rightPupilX, int rightPupilY,
                  bool inBlink);

    // Append an exercise result row.
    void logExerciseResult(const ExerciseResult &result);

    // Write a final summary section and flush the file.
    void writeReport(const HealthSummary &summary,
                     const std::vector<ExerciseResult> &results);

    // Returns the path of the CSV file being written.
    std::string getFilePath() const;

private:
    std::ofstream file;
    std::string   filePath;
};

#endif
