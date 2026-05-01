#include <ctime>
#include <iomanip>
#include <sstream>
#include <iostream>

#include "sessionLogger.h"

// Build a filename like:  session_2026-05-01_22-32-42.csv
static std::string makeFilename() {
    std::time_t now = std::time(nullptr);
    struct tm *lt   = std::localtime(&now);
    std::ostringstream oss;
    oss << "session_"
        << std::setw(4) << std::setfill('0') << (lt->tm_year + 1900) << "-"
        << std::setw(2) << std::setfill('0') << (lt->tm_mon  + 1)    << "-"
        << std::setw(2) << std::setfill('0') <<  lt->tm_mday          << "_"
        << std::setw(2) << std::setfill('0') <<  lt->tm_hour          << "-"
        << std::setw(2) << std::setfill('0') <<  lt->tm_min           << "-"
        << std::setw(2) << std::setfill('0') <<  lt->tm_sec
        << ".csv";
    return oss.str();
}

SessionLogger::SessionLogger() {
    filePath = makeFilename();
    file.open(filePath.c_str());
    if (!file.is_open()) {
        std::cerr << "[SessionLogger] Warning: could not open " << filePath << "\n";
        return;
    }
    // Header for frame rows
    file << "# eyeLike session log\n";
    file << "timestamp_s,left_x,left_y,right_x,right_y,in_blink\n";
}

SessionLogger::~SessionLogger() {
    if (file.is_open()) file.close();
}

void SessionLogger::logFrame(double timestamp,
                              int leftPupilX,  int leftPupilY,
                              int rightPupilX, int rightPupilY,
                              bool inBlink) {
    if (!file.is_open()) return;
    file << std::fixed << std::setprecision(4)
         << timestamp << ","
         << leftPupilX  << "," << leftPupilY  << ","
         << rightPupilX << "," << rightPupilY << ","
         << (inBlink ? 1 : 0) << "\n";
}

void SessionLogger::logExerciseResult(const ExerciseResult &r) {
    if (!file.is_open()) return;
    file << "# EXERCISE," << r.name
         << ",score," << std::fixed << std::setprecision(1) << r.score
         << ",frames_total," << r.framesTotal
         << ",frames_on_target," << r.framesOnTarget
         << ",feedback," << r.feedback << "\n";
}

void SessionLogger::writeReport(const HealthSummary &summary,
                                 const std::vector<ExerciseResult> &results) {
    if (!file.is_open()) return;

    file << "\n# ===== SESSION HEALTH REPORT =====\n";
    file << "# Blink count: "          << summary.blinks.totalBlinks    << "\n";
    file << "# Blinks per minute: "    << std::fixed << std::setprecision(1)
         << summary.blinks.blinksPerMinute                               << "\n";
    file << "# Avg blink duration ms: "<< std::fixed << std::setprecision(1)
         << summary.blinks.avgBlinkDurationMs                            << "\n";
    file << "# Low blink rate flag: "  << (summary.blinks.lowBlinkRate  ? "YES" : "no") << "\n";
    file << "# High blink rate flag: " << (summary.blinks.highBlinkRate ? "YES" : "no") << "\n";
    file << "# Pupil asymmetry flag: " << (summary.symmetry.asymmetryDetected ? "YES" : "no") << "\n";
    file << "# Nystagmus flag: "       << (summary.stability.nystagmusDetected ? "YES" : "no") << "\n";
    file << "# Gaze variance X: "      << std::fixed << std::setprecision(2)
         << summary.stability.varianceX                                  << "\n";
    file << "# Gaze variance Y: "      << std::fixed << std::setprecision(2)
         << summary.stability.varianceY                                  << "\n";

    file << "# --- Advice ---\n";
    for (const std::string &a : summary.advice)
        file << "#   " << a << "\n";

    if (!results.empty()) {
        file << "# --- Exercise Results ---\n";
        for (const ExerciseResult &r : results)
            file << "#   " << r.name
                 << "  score=" << std::fixed << std::setprecision(1)
                 << r.score << "%\n";
    }

    file.flush();
    std::cout << "[SessionLogger] Report saved to: " << filePath << "\n";
}

std::string SessionLogger::getFilePath() const {
    return filePath;
}
