// Shared SRT subtitle formatting for FunASR ggml runtimes.
#pragma once
#include <cstdio>
#include <string>

static void format_srt_timestamp(int ms, char * buf, size_t buf_size) {
    if (ms < 0) ms = 0;
    int total_sec = ms / 1000;
    int rem_ms    = ms % 1000;
    int sec       = total_sec % 60;
    int total_min = total_sec / 60;
    int min       = total_min % 60;
    int hours     = total_min / 60;
    snprintf(buf, buf_size, "%02d:%02d:%02d,%03d", hours, min, sec, rem_ms);
}

// Print a complete SRT entry: index, timestamp range, text, blank separator.
static void format_srt_line(int srt_idx, int start_ms, int end_ms, const std::string & text) {
    char ts0[32], ts1[32];
    format_srt_timestamp(start_ms, ts0, sizeof(ts0));
    format_srt_timestamp(end_ms,   ts1, sizeof(ts1));
    printf("%d\n%s --> %s\n%s\n\n", srt_idx, ts0, ts1, text.c_str());
}
