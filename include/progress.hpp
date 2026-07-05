#pragma once
#include "core_types.hpp"
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <algorithm>
#include <cstdio>

#// Enable Windows console UTF-8 support if on Windows
#if defined(_WIN32) || defined(_WIN64)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

class ProgressIndicator {
public:
    explicit ProgressIndicator(int total, std::string label = "")
        : total_(total), completed_(0), last_printed_width_(0), label_(std::move(label)) {
        start_time_ = std::chrono::steady_clock::now();

        // Fix Windows encoding for Unicode blocks
#if defined(_WIN32) || defined(_WIN64)
        SetConsoleOutputCP(CP_UTF8);
#endif
    }

    void tick(int n = 1) {
        int done = completed_.fetch_add(n) + n;
        // Throttle: update roughly every 0.5-1% or at the end
        int threshold = std::max(1, total_ / 200);

        // Final tick always forces a display
        if (done >= total_) {
            display(done);
            return;
        }

        // Throttle check
        if (done % threshold < n) {
            // Use try_lock to prevent thread-piling and stuttering
            if (mtx_.try_lock()) {
                // Double check progress hasn't regressed behind a newer thread's print
                display(completed_.load());
                mtx_.unlock();
            }
        }
    }

    void finish() {
        completed_ = total_;
        display(total_, true);
        printf("\n");
    }

private:
    void display(int done, bool final = false) {
        // Fallback to lock_guard if called from finish() where we absolutely must block
        std::unique_lock<std::mutex> lock(mtx_, std::defer_lock);
        if (!final) {
            // If called from finish(), it's already locked or needs a hard lock
            lock.lock();
        }

        // Ensure we don't divide by zero or overshoot boundaries
        done = std::min(done, total_);
        double progress = total_ > 0 ? static_cast<double>(done) / total_ : 1.0;

        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time_).count();

        // Clear previous line
        printf("\r%*s\r", last_printed_width_, "");

        // Build bar safely across platforms
        constexpr int bar_width = 25;
        int filled = static_cast<int>(bar_width * progress);

        std::string bar = "[";
        for (int i = 0; i < bar_width; ++i) {
            // Using standard UTF-8 blocks works on Windows now thanks to SetConsoleOutputCP
            bar += (i < filled) ? "█" : "░";
        }
        bar += "]";

        // Time info
        std::string time_info;
        if (final) {
            time_info = fmt_time(elapsed);
        } else if (progress > 0.005) {
            double eta = elapsed * (1.0 - progress) / progress;
            time_info = "ETA " + fmt_time(eta);
        } else {
            time_info = "ETA --:--";
        }

        // Format and print string safely
        char buf[512];
        if (!label_.empty()) {
            snprintf(buf, sizeof(buf), " %s %s %d/%d (%3d%%) %s",
                     label_.c_str(), bar.c_str(), done, total_,
                     static_cast<int>(progress * 100), time_info.c_str());
        } else {
            snprintf(buf, sizeof(buf), " %s %d/%d (%3d%%) %s",
                     bar.c_str(), done, total_,
                     static_cast<int>(progress * 100), time_info.c_str());
        }

        printf("%s", buf);
        fflush(stdout);
        last_printed_width_ = static_cast<int>(std::string(buf).size());
    }

    static std::string fmt_time(double seconds) {
        if (seconds < 0) return "--:--";
        int t = static_cast<int>(seconds);
        if (t >= 3600) {
            return std::to_string(t / 3600) + ":" +
                   pad2((t % 3600) / 60) + ":" + pad2(t % 60);
        }
        return std::to_string(t / 60) + ":" + pad2(t % 60);
    }

    static std::string pad2(int n) {
        char buf[8];
        snprintf(buf, sizeof(buf), "%02d", n);
        return buf;
    }

    int total_;
    std::atomic<int> completed_;
    std::chrono::steady_clock::time_point start_time_;
    std::mutex mtx_;
    int last_printed_width_;
    std::string label_;
};
