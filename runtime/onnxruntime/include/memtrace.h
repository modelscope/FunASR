/**
 * VmRSS + NDJSON tracing for memory debugging (FUNASR_ENABLE_MEMDBG=1).
 */
#pragma once

#include <cstdint>

namespace funasr {

uint64_t MemtraceVmRSSKb();

int64_t MemtraceNextDecoderId();

void MemtraceSetTlsTraceId(int64_t id);

void MemtraceClearTlsTraceId();

int64_t MemtraceGetTlsTraceId();

void MemtraceLog(const char* phase, const char* hypothesisId, int64_t trace_id,
                 long long data_a = 0, long long data_b = 0);

struct MemtraceDecoderSession {
  int64_t trace_id_;

  MemtraceDecoderSession(long long buffer_bytes, long long is_final_flag);
  ~MemtraceDecoderSession();

  int64_t trace_id() const { return trace_id_; }
};

}  // namespace funasr
