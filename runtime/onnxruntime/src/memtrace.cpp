#include "memtrace.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace funasr {
namespace {

std::mutex g_memtrace_mu;
std::atomic<int64_t> g_decoder_seq{0};

thread_local int64_t g_tls_trace_id = -1;
thread_local uint64_t g_tls_start_rss_kb = 0;

static FILE* OpenMemtraceLog() {
  const char* e = std::getenv("FUNASR_DEBUG_LOG");
  if (e && e[0]) {
    FILE* fp = std::fopen(e, "a");
    if (fp) return fp;
  }
  FILE* fp2 = std::fopen("/data/project/liuyuntao/FUN_ASR/.cursor/debug-08e994.log", "a");
  if (fp2) return fp2;
  return std::fopen("/tmp/funasr_debug_08e994.ndjson", "a");
}

static uint64_t ReadVmRSSKb() {
  FILE* f = std::fopen("/proc/self/status", "r");
  if (!f) return 0;
  char line[256];
  unsigned long kb = 0;
  while (std::fgets(line, sizeof(line), f)) {
    if (std::strncmp(line, "VmRSS:", 6) == 0) {
      if (std::sscanf(line, "VmRSS: %lu kB", &kb) == 1) break;
    }
  }
  std::fclose(f);
  return static_cast<uint64_t>(kb);
}

}  // namespace

uint64_t MemtraceVmRSSKb() { return ReadVmRSSKb(); }

int64_t MemtraceNextDecoderId() { return ++g_decoder_seq; }

void MemtraceSetTlsTraceId(int64_t id) {
  g_tls_trace_id = id;
  g_tls_start_rss_kb = ReadVmRSSKb();
}

void MemtraceClearTlsTraceId() {
  g_tls_trace_id = -1;
  g_tls_start_rss_kb = 0;
}

int64_t MemtraceGetTlsTraceId() { return g_tls_trace_id; }

void MemtraceLog(const char* phase, const char* hypothesisId, int64_t trace_id,
                 long long data_a, long long data_b) {
  const char* en = std::getenv("FUNASR_ENABLE_MEMDBG");
  if (!en || en[0] != '1') return;

  uint64_t rss = ReadVmRSSKb();
  long long delta = 0;
  if (trace_id >= 0 && g_tls_trace_id == trace_id && g_tls_start_rss_kb > 0)
    delta = static_cast<long long>(rss) - static_cast<long long>(g_tls_start_rss_kb);

  std::lock_guard<std::mutex> lk(g_memtrace_mu);
  FILE* fp = OpenMemtraceLog();
  if (!fp) return;
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count();
  std::fprintf(fp,
               "{\"sessionId\":\"08e994\",\"hypothesisId\":\"%s\",\"location\":\"memtrace\",\"message\":\"%s\","
               "\"data\":{\"trace_id\":%lld,\"rss_kb\":%llu,\"delta_from_decoder_start_kb\":%lld,"
               "\"data_a\":%lld,\"data_b\":%lld},\"timestamp\":%lld}\n",
               hypothesisId, phase, (long long)trace_id, (unsigned long long)rss, delta, data_a, data_b,
               (long long)ms);
  std::fclose(fp);
}

MemtraceDecoderSession::MemtraceDecoderSession(long long buffer_bytes, long long is_final_flag) {
  trace_id_ = MemtraceNextDecoderId();
  MemtraceSetTlsTraceId(trace_id_);
  MemtraceLog("ws_do_decoder_start", "WS0", trace_id_, buffer_bytes, is_final_flag);
}

MemtraceDecoderSession::~MemtraceDecoderSession() {
  MemtraceLog("ws_do_decoder_end", "WS9", trace_id_, 0, 0);
  MemtraceClearTlsTraceId();
}

}  // namespace funasr
