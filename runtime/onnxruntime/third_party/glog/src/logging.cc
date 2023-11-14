// Copyright (c) 1999, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#define _GNU_SOURCE 1 // needed for O_NOFOLLOW and pread()/pwrite()

#include "utilities.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iomanip>
#include <iterator>
#include <string>

#ifdef HAVE_UNISTD_H
# include <unistd.h>  // For _exit.
#endif
#include <climits>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef HAVE_SYS_UTSNAME_H
# include <sys/utsname.h>  // For uname.
#endif
#include <ctime>
#include <fcntl.h>
#include <cstdio>
#include <iostream>
#include <cstdarg>
#include <cstdlib>
#ifdef HAVE_PWD_H
# include <pwd.h>
#endif
#ifdef HAVE_SYSLOG_H
# include <syslog.h>
#endif
#ifdef HAVE__CHSIZE_S
#include <io.h> // for truncate log file
#endif
#include <vector>
#include <cerrno>                   // for errno
#include <sstream>
#include <regex>
#include <cctype> // for std::isspace
#ifdef GLOG_OS_WINDOWS
#include "windows/dirent.h"
#else
#include <dirent.h> // for automatic removal of old logs
#endif
#include "base/commandlineflags.h"  // to get the program name
#include "base/googleinit.h"
#include "glog/logging.h"
#include "glog/raw_logging.h"

#ifdef HAVE_STACKTRACE
# include "stacktrace.h"
#endif

#ifdef __ANDROID__
#include <android/log.h>
#endif

using std::string;
using std::vector;
using std::setw;
using std::setfill;
using std::hex;
using std::dec;
using std::min;
using std::ostream;
using std::ostringstream;

using std::FILE;
using std::fwrite;
using std::fclose;
using std::fflush;
using std::fprintf;
using std::perror;

#ifdef __QNX__
using std::fdopen;
#endif

#ifdef _WIN32
#define fdopen _fdopen
#endif

// There is no thread annotation support.
#define EXCLUSIVE_LOCKS_REQUIRED(mu)

static bool BoolFromEnv(const char *varname, bool defval) {
  const char* const valstr = getenv(varname);
  if (!valstr) {
    return defval;
  }
  return memchr("tTyY1\0", valstr[0], 6) != nullptr;
}

GLOG_DEFINE_bool(timestamp_in_logfile_name,
                 BoolFromEnv("GOOGLE_TIMESTAMP_IN_LOGFILE_NAME", true),
                 "put a timestamp at the end of the log file name");
GLOG_DEFINE_bool(logtostderr, BoolFromEnv("GOOGLE_LOGTOSTDERR", false),
                 "log messages go to stderr instead of logfiles");
GLOG_DEFINE_bool(alsologtostderr, BoolFromEnv("GOOGLE_ALSOLOGTOSTDERR", false),
                 "log messages go to stderr in addition to logfiles");
GLOG_DEFINE_bool(colorlogtostderr, false,
                 "color messages logged to stderr (if supported by terminal)");
GLOG_DEFINE_bool(colorlogtostdout, false,
                 "color messages logged to stdout (if supported by terminal)");
GLOG_DEFINE_bool(logtostdout, BoolFromEnv("GOOGLE_LOGTOSTDOUT", false),
                 "log messages go to stdout instead of logfiles");
#ifdef GLOG_OS_LINUX
GLOG_DEFINE_bool(drop_log_memory, true, "Drop in-memory buffers of log contents. "
                 "Logs can grow very quickly and they are rarely read before they "
                 "need to be evicted from memory. Instead, drop them from memory "
                 "as soon as they are flushed to disk.");
#endif

// By default, errors (including fatal errors) get logged to stderr as
// well as the file.
//
// The default is ERROR instead of FATAL so that users can see problems
// when they run a program without having to look in another file.
DEFINE_int32(stderrthreshold,
             GOOGLE_NAMESPACE::GLOG_ERROR,
             "log messages at or above this level are copied to stderr in "
             "addition to logfiles.  This flag obsoletes --alsologtostderr.");

GLOG_DEFINE_string(alsologtoemail, "",
                   "log messages go to these email addresses "
                   "in addition to logfiles");
GLOG_DEFINE_bool(log_file_header, true,
                 "Write the file header at the start of each log file");
GLOG_DEFINE_bool(log_prefix, true,
                 "Prepend the log prefix to the start of each log line");
GLOG_DEFINE_bool(log_year_in_prefix, true,
                 "Include the year in the log prefix");
GLOG_DEFINE_int32(minloglevel, 0, "Messages logged at a lower level than this don't "
                  "actually get logged anywhere");
GLOG_DEFINE_int32(logbuflevel, 0,
                  "Buffer log messages logged at this level or lower"
                  " (-1 means don't buffer; 0 means buffer INFO only;"
                  " ...)");
GLOG_DEFINE_int32(logbufsecs, 30,
                  "Buffer log messages for at most this many seconds");

GLOG_DEFINE_int32(logcleansecs, 60 * 5, // every 5 minutes
                  "Clean overdue logs every this many seconds");

GLOG_DEFINE_int32(logemaillevel, 999,
                  "Email log messages logged at this level or higher"
                  " (0 means email all; 3 means email FATAL only;"
                  " ...)");
GLOG_DEFINE_string(logmailer, "",
                   "Mailer used to send logging email");

// Compute the default value for --log_dir
static const char* DefaultLogDir() {
  const char* env;
  env = getenv("GOOGLE_LOG_DIR");
  if (env != nullptr && env[0] != '\0') {
    return env;
  }
  env = getenv("TEST_TMPDIR");
  if (env != nullptr && env[0] != '\0') {
    return env;
  }
  return "";
}

GLOG_DEFINE_int32(logfile_mode, 0664, "Log file mode/permissions.");

GLOG_DEFINE_string(log_dir, DefaultLogDir(),
                   "If specified, logfiles are written into this directory instead "
                   "of the default logging directory.");
GLOG_DEFINE_string(log_link, "", "Put additional links to the log "
                   "files in this directory");

GLOG_DEFINE_uint32(max_log_size, 1800,
                   "approx. maximum log file size (in MB). A value of 0 will "
                   "be silently overridden to 1.");

GLOG_DEFINE_bool(stop_logging_if_full_disk, false,
                 "Stop attempting to log to disk if the disk is full.");

GLOG_DEFINE_string(log_backtrace_at, "",
                   "Emit a backtrace when logging at file:linenum.");

GLOG_DEFINE_bool(log_utc_time, false,
    "Use UTC time for logging.");

// TODO(hamaji): consider windows
enum { PATH_SEPARATOR = '/' };

#ifndef HAVE_PREAD
#if defined(GLOG_OS_WINDOWS)
#include <basetsd.h>
#define ssize_t SSIZE_T
#endif
static ssize_t pread(int fd, void* buf, size_t count, off_t offset) {
  off_t orig_offset = lseek(fd, 0, SEEK_CUR);
  if (orig_offset == (off_t)-1)
    return -1;
  if (lseek(fd, offset, SEEK_CUR) == (off_t)-1)
    return -1;
  ssize_t len = read(fd, buf, count);
  if (len < 0)
    return len;
  if (lseek(fd, orig_offset, SEEK_SET) == (off_t)-1)
    return -1;
  return len;
}
#endif  // !HAVE_PREAD

#ifndef HAVE_PWRITE
static ssize_t pwrite(int fd, void* buf, size_t count, off_t offset) {
  off_t orig_offset = lseek(fd, 0, SEEK_CUR);
  if (orig_offset == (off_t)-1)
    return -1;
  if (lseek(fd, offset, SEEK_CUR) == (off_t)-1)
    return -1;
  ssize_t len = write(fd, buf, count);
  if (len < 0)
    return len;
  if (lseek(fd, orig_offset, SEEK_SET) == (off_t)-1)
    return -1;
  return len;
}
#endif  // !HAVE_PWRITE

static void GetHostName(string* hostname) {
#if defined(HAVE_SYS_UTSNAME_H)
  struct utsname buf;
  if (uname(&buf) < 0) {
    // ensure null termination on failure
    *buf.nodename = '\0';
  }
  *hostname = buf.nodename;
#elif defined(GLOG_OS_WINDOWS)
  char buf[MAX_COMPUTERNAME_LENGTH + 1];
  DWORD len = MAX_COMPUTERNAME_LENGTH + 1;
  if (GetComputerNameA(buf, &len)) {
    *hostname = buf;
  } else {
    hostname->clear();
  }
#else
# warning There is no way to retrieve the host name.
  *hostname = "(unknown)";
#endif
}

// Returns true iff terminal supports using colors in output.
static bool TerminalSupportsColor() {
  bool term_supports_color = false;
#ifdef GLOG_OS_WINDOWS
  // on Windows TERM variable is usually not set, but the console does
  // support colors.
  term_supports_color = true;
#else
  // On non-Windows platforms, we rely on the TERM variable.
  const char* const term = getenv("TERM");
  if (term != nullptr && term[0] != '\0') {
    term_supports_color =
      !strcmp(term, "xterm") ||
      !strcmp(term, "xterm-color") ||
      !strcmp(term, "xterm-256color") ||
      !strcmp(term, "screen-256color") ||
      !strcmp(term, "konsole") ||
      !strcmp(term, "konsole-16color") ||
      !strcmp(term, "konsole-256color") ||
      !strcmp(term, "screen") ||
      !strcmp(term, "linux") ||
      !strcmp(term, "cygwin");
  }
#endif
  return term_supports_color;
}

_START_GOOGLE_NAMESPACE_

enum GLogColor {
  COLOR_DEFAULT,
  COLOR_RED,
  COLOR_GREEN,
  COLOR_YELLOW
};

static GLogColor SeverityToColor(LogSeverity severity) {
  assert(severity >= 0 && severity < NUM_SEVERITIES);
  GLogColor color = COLOR_DEFAULT;
  switch (severity) {
  case GLOG_INFO:
    color = COLOR_DEFAULT;
    break;
  case GLOG_WARNING:
    color = COLOR_YELLOW;
    break;
  case GLOG_ERROR:
  case GLOG_FATAL:
    color = COLOR_RED;
    break;
  default:
    // should never get here.
    assert(false);
  }
  return color;
}

#ifdef GLOG_OS_WINDOWS

// Returns the character attribute for the given color.
static WORD GetColorAttribute(GLogColor color) {
  switch (color) {
    case COLOR_RED:    return FOREGROUND_RED;
    case COLOR_GREEN:  return FOREGROUND_GREEN;
    case COLOR_YELLOW: return FOREGROUND_RED | FOREGROUND_GREEN;
    default:           return 0;
  }
}

#else

// Returns the ANSI color code for the given color.
static const char* GetAnsiColorCode(GLogColor color) {
  switch (color) {
  case COLOR_RED:     return "1";
  case COLOR_GREEN:   return "2";
  case COLOR_YELLOW:  return "3";
  case COLOR_DEFAULT:  return "";
  };
  return nullptr;  // stop warning about return type.
}

#endif  // GLOG_OS_WINDOWS

// Safely get max_log_size, overriding to 1 if it somehow gets defined as 0
static uint32 MaxLogSize() {
  return (FLAGS_max_log_size > 0 && FLAGS_max_log_size < 4096
              ? FLAGS_max_log_size
              : 1);
}

// An arbitrary limit on the length of a single log message.  This
// is so that streaming can be done more efficiently.
const size_t LogMessage::kMaxLogMessageLen = 30000;

struct LogMessage::LogMessageData  {
  LogMessageData();

  int preserved_errno_;      // preserved errno
  // Buffer space; contains complete message text.
  char message_text_[LogMessage::kMaxLogMessageLen+1];
  LogStream stream_;
  char severity_;      // What level is this LogMessage logged at?
  int line_;                 // line number where logging call is.
  void (LogMessage::*send_method_)();  // Call this in destructor to send
  union {  // At most one of these is used: union to keep the size low.
    LogSink* sink_;  // nullptr or sink to send message to
    std::vector<std::string>*
        outvec_;            // nullptr or vector to push message onto
    std::string* message_;  // nullptr or string to write message into
  };
  size_t num_prefix_chars_;     // # of chars of prefix in this message
  size_t num_chars_to_log_;     // # of chars of msg to send to log
  size_t num_chars_to_syslog_;  // # of chars of msg to send to syslog
  const char* basename_;        // basename of file that called LOG
  const char* fullname_;        // fullname of file that called LOG
  bool has_been_flushed_;       // false => data has not been flushed
  bool first_fatal_;            // true => this was first fatal msg

 private:
  LogMessageData(const LogMessageData&) = delete;
  void operator=(const LogMessageData&) = delete;
};

// A mutex that allows only one thread to log at a time, to keep things from
// getting jumbled.  Some other very uncommon logging operations (like
// changing the destination file for log messages of a given severity) also
// lock this mutex.  Please be sure that anybody who might possibly need to
// lock it does so.
static Mutex log_mutex;

// Number of messages sent at each severity.  Under log_mutex.
int64 LogMessage::num_messages_[NUM_SEVERITIES] = {0, 0, 0, 0};

// Globally disable log writing (if disk is full)
static bool stop_writing = false;

const char*const LogSeverityNames[NUM_SEVERITIES] = {
  "INFO", "WARNING", "ERROR", "FATAL"
};

// Has the user called SetExitOnDFatal(true)?
static bool exit_on_dfatal = true;

const char* GetLogSeverityName(LogSeverity severity) {
  return LogSeverityNames[severity];
}

static bool SendEmailInternal(const char*dest, const char *subject,
                              const char*body, bool use_logging);

base::Logger::~Logger() = default;

namespace  {
  // Optional user-configured callback to print custom prefixes.
CustomPrefixCallback custom_prefix_callback = nullptr;
// User-provided data to pass to the callback:
void* custom_prefix_callback_data = nullptr;
}

namespace {

// Encapsulates all file-system related state
class LogFileObject : public base::Logger {
 public:
  LogFileObject(LogSeverity severity, const char* base_filename);
  ~LogFileObject() override;

  void Write(bool force_flush,  // Should we force a flush here?
             time_t timestamp,  // Timestamp for this entry
             const char* message, size_t message_len) override;

  // Configuration options
  void SetBasename(const char* basename);
  void SetExtension(const char* ext);
  void SetSymlinkBasename(const char* symlink_basename);

  // Normal flushing routine
  void Flush() override;

  // It is the actual file length for the system loggers,
  // i.e., INFO, ERROR, etc.
  uint32 LogSize() override {
    MutexLock l(&lock_);
    return file_length_;
  }

  // Internal flush routine.  Exposed so that FlushLogFilesUnsafe()
  // can avoid grabbing a lock.  Usually Flush() calls it after
  // acquiring lock_.
  void FlushUnlocked();

 private:
  static const uint32 kRolloverAttemptFrequency = 0x20;

  Mutex lock_;
  bool base_filename_selected_;
  string base_filename_;
  string symlink_basename_;
  string filename_extension_;     // option users can specify (eg to add port#)
  FILE* file_{nullptr};
  LogSeverity severity_;
  uint32 bytes_since_flush_{0};
  uint32 dropped_mem_length_{0};
  uint32 file_length_{0};
  unsigned int rollover_attempt_;
  int64 next_flush_time_{0};  // cycle count at which to flush log
  WallTime start_time_;

  // Actually create a logfile using the value of base_filename_ and the
  // optional argument time_pid_string
  // REQUIRES: lock_ is held
  bool CreateLogfile(const string& time_pid_string);
};

// Encapsulate all log cleaner related states
class LogCleaner {
 public:
  LogCleaner();

  // Setting overdue_days to 0 days will delete all logs.
  void Enable(unsigned int overdue_days);
  void Disable();

  // update next_cleanup_time_
  void UpdateCleanUpTime();

  void Run(bool base_filename_selected,
           const string& base_filename,
           const string& filename_extension);

  bool enabled() const { return enabled_; }

 private:
  vector<string> GetOverdueLogNames(string log_directory, unsigned int days,
                                    const string& base_filename,
                                    const string& filename_extension) const;

  bool IsLogFromCurrentProject(const string& filepath,
                               const string& base_filename,
                               const string& filename_extension) const;

  bool IsLogLastModifiedOver(const string& filepath, unsigned int days) const;

  bool enabled_{false};
  unsigned int overdue_days_{7};
  int64 next_cleanup_time_{0};  // cycle count at which to clean overdue log
};

LogCleaner log_cleaner;

}  // namespace

class LogDestination {
 public:
  friend class LogMessage;
  friend void ReprintFatalMessage();
  friend base::Logger* base::GetLogger(LogSeverity);
  friend void base::SetLogger(LogSeverity, base::Logger*);

  // These methods are just forwarded to by their global versions.
  static void SetLogDestination(LogSeverity severity,
				const char* base_filename);
  static void SetLogSymlink(LogSeverity severity,
                            const char* symlink_basename);
  static void AddLogSink(LogSink *destination);
  static void RemoveLogSink(LogSink *destination);
  static void SetLogFilenameExtension(const char* filename_extension);
  static void SetStderrLogging(LogSeverity min_severity);
  static void SetEmailLogging(LogSeverity min_severity, const char* addresses);
  static void LogToStderr();
  // Flush all log files that are at least at the given severity level
  static void FlushLogFiles(int min_severity);
  static void FlushLogFilesUnsafe(int min_severity);

  // we set the maximum size of our packet to be 1400, the logic being
  // to prevent fragmentation.
  // Really this number is arbitrary.
  static const int kNetworkBytes = 1400;

  static const string& hostname();
  static const bool& terminal_supports_color() {
    return terminal_supports_color_;
  }

  static void DeleteLogDestinations();

 private:
  LogDestination(LogSeverity severity, const char* base_filename);
  ~LogDestination();

  // Take a log message of a particular severity and log it to stderr
  // iff it's of a high enough severity to deserve it.
  static void MaybeLogToStderr(LogSeverity severity, const char* message,
			       size_t message_len, size_t prefix_len);

  // Take a log message of a particular severity and log it to email
  // iff it's of a high enough severity to deserve it.
  static void MaybeLogToEmail(LogSeverity severity, const char* message,
			      size_t len);
  // Take a log message of a particular severity and log it to a file
  // iff the base filename is not "" (which means "don't log to me")
  static void MaybeLogToLogfile(LogSeverity severity,
                                time_t timestamp,
				const char* message, size_t len);
  // Take a log message of a particular severity and log it to the file
  // for that severity and also for all files with severity less than
  // this severity.
  static void LogToAllLogfiles(LogSeverity severity,
                               time_t timestamp,
                               const char* message, size_t len);

  // Send logging info to all registered sinks.
  static void LogToSinks(LogSeverity severity, const char* full_filename,
                         const char* base_filename, int line,
                         const LogMessageTime& logmsgtime, const char* message,
                         size_t message_len);

  // Wait for all registered sinks via WaitTillSent
  // including the optional one in "data".
  static void WaitForSinks(LogMessage::LogMessageData* data);

  static LogDestination* log_destination(LogSeverity severity);

  base::Logger* GetLoggerImpl() const { return logger_; }
  void SetLoggerImpl(base::Logger* logger);
  void ResetLoggerImpl() { SetLoggerImpl(&fileobject_); }

  LogFileObject fileobject_;
  base::Logger* logger_;      // Either &fileobject_, or wrapper around it

  static LogDestination* log_destinations_[NUM_SEVERITIES];
  static LogSeverity email_logging_severity_;
  static string addresses_;
  static string hostname_;
  static bool terminal_supports_color_;

  // arbitrary global logging destinations.
  static vector<LogSink*>* sinks_;

  // Protects the vector sinks_,
  // but not the LogSink objects its elements reference.
  static Mutex sink_mutex_;

  // Disallow
  LogDestination(const LogDestination&) = delete;
  LogDestination& operator=(const LogDestination&) = delete;
};

// Errors do not get logged to email by default.
LogSeverity LogDestination::email_logging_severity_ = 99999;

string LogDestination::addresses_;
string LogDestination::hostname_;

vector<LogSink*>* LogDestination::sinks_ = nullptr;
Mutex LogDestination::sink_mutex_;
bool LogDestination::terminal_supports_color_ = TerminalSupportsColor();

/* static */
const string& LogDestination::hostname() {
  if (hostname_.empty()) {
    GetHostName(&hostname_);
    if (hostname_.empty()) {
      hostname_ = "(unknown)";
    }
  }
  return hostname_;
}

LogDestination::LogDestination(LogSeverity severity,
                               const char* base_filename)
  : fileobject_(severity, base_filename),
    logger_(&fileobject_) {
}

LogDestination::~LogDestination() {
  ResetLoggerImpl();
}

void LogDestination::SetLoggerImpl(base::Logger* logger) {
  if (logger_ == logger) {
    // Prevent releasing currently held sink on reset
    return;
  }

  if (logger_ && logger_ != &fileobject_) {
    // Delete user-specified logger set via SetLogger().
    delete logger_;
  }
  logger_ = logger;
}

inline void LogDestination::FlushLogFilesUnsafe(int min_severity) {
  // assume we have the log_mutex or we simply don't care
  // about it
  for (int i = min_severity; i < NUM_SEVERITIES; i++) {
    LogDestination* log = log_destinations_[i];
    if (log != nullptr) {
      // Flush the base fileobject_ logger directly instead of going
      // through any wrappers to reduce chance of deadlock.
      log->fileobject_.FlushUnlocked();
    }
  }
}

inline void LogDestination::FlushLogFiles(int min_severity) {
  // Prevent any subtle race conditions by wrapping a mutex lock around
  // all this stuff.
  MutexLock l(&log_mutex);
  for (int i = min_severity; i < NUM_SEVERITIES; i++) {
    LogDestination* log = log_destination(i);
    if (log != nullptr) {
      log->logger_->Flush();
    }
  }
}

inline void LogDestination::SetLogDestination(LogSeverity severity,
					      const char* base_filename) {
  assert(severity >= 0 && severity < NUM_SEVERITIES);
  // Prevent any subtle race conditions by wrapping a mutex lock around
  // all this stuff.
  MutexLock l(&log_mutex);
  log_destination(severity)->fileobject_.SetBasename(base_filename);
}

inline void LogDestination::SetLogSymlink(LogSeverity severity,
                                          const char* symlink_basename) {
  CHECK_GE(severity, 0);
  CHECK_LT(severity, NUM_SEVERITIES);
  MutexLock l(&log_mutex);
  log_destination(severity)->fileobject_.SetSymlinkBasename(symlink_basename);
}

inline void LogDestination::AddLogSink(LogSink *destination) {
  // Prevent any subtle race conditions by wrapping a mutex lock around
  // all this stuff.
  MutexLock l(&sink_mutex_);
  if (!sinks_)  sinks_ = new vector<LogSink*>;
  sinks_->push_back(destination);
}

inline void LogDestination::RemoveLogSink(LogSink *destination) {
  // Prevent any subtle race conditions by wrapping a mutex lock around
  // all this stuff.
  MutexLock l(&sink_mutex_);
  // This doesn't keep the sinks in order, but who cares?
  if (sinks_) {
    sinks_->erase(std::remove(sinks_->begin(), sinks_->end(), destination), sinks_->end());
  }
}

inline void LogDestination::SetLogFilenameExtension(const char* ext) {
  // Prevent any subtle race conditions by wrapping a mutex lock around
  // all this stuff.
  MutexLock l(&log_mutex);
  for ( int severity = 0; severity < NUM_SEVERITIES; ++severity ) {
    log_destination(severity)->fileobject_.SetExtension(ext);
  }
}

inline void LogDestination::SetStderrLogging(LogSeverity min_severity) {
  assert(min_severity >= 0 && min_severity < NUM_SEVERITIES);
  // Prevent any subtle race conditions by wrapping a mutex lock around
  // all this stuff.
  MutexLock l(&log_mutex);
  FLAGS_stderrthreshold = min_severity;
}

inline void LogDestination::LogToStderr() {
  // *Don't* put this stuff in a mutex lock, since SetStderrLogging &
  // SetLogDestination already do the locking!
  SetStderrLogging(0);            // thus everything is "also" logged to stderr
  for ( int i = 0; i < NUM_SEVERITIES; ++i ) {
    SetLogDestination(i, "");     // "" turns off logging to a logfile
  }
}

inline void LogDestination::SetEmailLogging(LogSeverity min_severity,
					    const char* addresses) {
  assert(min_severity >= 0 && min_severity < NUM_SEVERITIES);
  // Prevent any subtle race conditions by wrapping a mutex lock around
  // all this stuff.
  MutexLock l(&log_mutex);
  LogDestination::email_logging_severity_ = min_severity;
  LogDestination::addresses_ = addresses;
}

static void ColoredWriteToStderrOrStdout(FILE* output, LogSeverity severity,
                                         const char* message, size_t len) {
  bool is_stdout = (output == stdout);
  const GLogColor color = (LogDestination::terminal_supports_color() &&
                           ((!is_stdout && FLAGS_colorlogtostderr) ||
                            (is_stdout && FLAGS_colorlogtostdout)))
                              ? SeverityToColor(severity)
                              : COLOR_DEFAULT;

  // Avoid using cerr from this module since we may get called during
  // exit code, and cerr may be partially or fully destroyed by then.
  if (COLOR_DEFAULT == color) {
    fwrite(message, len, 1, output);
    return;
  }
#ifdef GLOG_OS_WINDOWS
  const HANDLE output_handle =
      GetStdHandle(is_stdout ? STD_OUTPUT_HANDLE : STD_ERROR_HANDLE);

  // Gets the current text color.
  CONSOLE_SCREEN_BUFFER_INFO buffer_info;
  GetConsoleScreenBufferInfo(output_handle, &buffer_info);
  const WORD old_color_attrs = buffer_info.wAttributes;

  // We need to flush the stream buffers into the console before each
  // SetConsoleTextAttribute call lest it affect the text that is already
  // printed but has not yet reached the console.
  fflush(output);
  SetConsoleTextAttribute(output_handle,
                          GetColorAttribute(color) | FOREGROUND_INTENSITY);
  fwrite(message, len, 1, output);
  fflush(output);
  // Restores the text color.
  SetConsoleTextAttribute(output_handle, old_color_attrs);
#else
  fprintf(output, "\033[0;3%sm", GetAnsiColorCode(color));
  fwrite(message, len, 1, output);
  fprintf(output, "\033[m");  // Resets the terminal to default.
#endif  // GLOG_OS_WINDOWS
}

static void ColoredWriteToStdout(LogSeverity severity, const char* message,
                                 size_t len) {
  FILE* output = stdout;
  // We also need to send logs to the stderr when the severity is
  // higher or equal to the stderr threshold.
  if (severity >= FLAGS_stderrthreshold) {
    output = stderr;
  }
  ColoredWriteToStderrOrStdout(output, severity, message, len);
}

static void ColoredWriteToStderr(LogSeverity severity, const char* message,
                                 size_t len) {
  ColoredWriteToStderrOrStdout(stderr, severity, message, len);
}

static void WriteToStderr(const char* message, size_t len) {
  // Avoid using cerr from this module since we may get called during
  // exit code, and cerr may be partially or fully destroyed by then.
  fwrite(message, len, 1, stderr);
}

inline void LogDestination::MaybeLogToStderr(LogSeverity severity,
					     const char* message, size_t message_len, size_t prefix_len) {
  if ((severity >= FLAGS_stderrthreshold) || FLAGS_alsologtostderr) {
    ColoredWriteToStderr(severity, message, message_len);
#ifdef GLOG_OS_WINDOWS
    (void) prefix_len;
    // On Windows, also output to the debugger
    ::OutputDebugStringA(message);
#elif defined(__ANDROID__)
    // On Android, also output to logcat
    const int android_log_levels[NUM_SEVERITIES] = {
      ANDROID_LOG_INFO,
      ANDROID_LOG_WARN,
      ANDROID_LOG_ERROR,
      ANDROID_LOG_FATAL,
    };
    __android_log_write(android_log_levels[severity],
                        glog_internal_namespace_::ProgramInvocationShortName(),
                        message + prefix_len);
#else
    (void) prefix_len;
#endif
  }
}


inline void LogDestination::MaybeLogToEmail(LogSeverity severity,
					    const char* message, size_t len) {
  if (severity >= email_logging_severity_ ||
      severity >= FLAGS_logemaillevel) {
    string to(FLAGS_alsologtoemail);
    if (!addresses_.empty()) {
      if (!to.empty()) {
        to += ",";
      }
      to += addresses_;
    }
    const string subject(string("[LOG] ") + LogSeverityNames[severity] + ": " +
                         glog_internal_namespace_::ProgramInvocationShortName());
    string body(hostname());
    body += "\n\n";
    body.append(message, len);

    // should NOT use SendEmail().  The caller of this function holds the
    // log_mutex and SendEmail() calls LOG/VLOG which will block trying to
    // acquire the log_mutex object.  Use SendEmailInternal() and set
    // use_logging to false.
    SendEmailInternal(to.c_str(), subject.c_str(), body.c_str(), false);
  }
}


inline void LogDestination::MaybeLogToLogfile(LogSeverity severity,
                                              time_t timestamp,
					      const char* message,
					      size_t len) {
  const bool should_flush = severity > FLAGS_logbuflevel;
  LogDestination* destination = log_destination(severity);
  destination->logger_->Write(should_flush, timestamp, message, len);
}

inline void LogDestination::LogToAllLogfiles(LogSeverity severity,
                                             time_t timestamp,
                                             const char* message,
                                             size_t len) {
  if (FLAGS_logtostdout) {  // global flag: never log to file
    ColoredWriteToStdout(severity, message, len);
  } else if (FLAGS_logtostderr) {  // global flag: never log to file
    ColoredWriteToStderr(severity, message, len);
  } else {
    for (int i = severity; i >= 0; --i) {
      LogDestination::MaybeLogToLogfile(i, timestamp, message, len);
    }
  }
}

inline void LogDestination::LogToSinks(LogSeverity severity,
                                       const char* full_filename,
                                       const char* base_filename, int line,
                                       const LogMessageTime& logmsgtime,
                                       const char* message,
                                       size_t message_len) {
  ReaderMutexLock l(&sink_mutex_);
  if (sinks_) {
    for (size_t i = sinks_->size(); i-- > 0; ) {
      (*sinks_)[i]->send(severity, full_filename, base_filename,
                         line, logmsgtime, message, message_len);
    }
  }
}

inline void LogDestination::WaitForSinks(LogMessage::LogMessageData* data) {
  ReaderMutexLock l(&sink_mutex_);
  if (sinks_) {
    for (size_t i = sinks_->size(); i-- > 0; ) {
      (*sinks_)[i]->WaitTillSent();
    }
  }
  const bool send_to_sink =
      (data->send_method_ == &LogMessage::SendToSink) ||
      (data->send_method_ == &LogMessage::SendToSinkAndLog);
  if (send_to_sink && data->sink_ != nullptr) {
    data->sink_->WaitTillSent();
  }
}

LogDestination* LogDestination::log_destinations_[NUM_SEVERITIES];

inline LogDestination* LogDestination::log_destination(LogSeverity severity) {
  assert(severity >=0 && severity < NUM_SEVERITIES);
  if (!log_destinations_[severity]) {
    log_destinations_[severity] = new LogDestination(severity, nullptr);
  }
  return log_destinations_[severity];
}

void LogDestination::DeleteLogDestinations() {
  for (auto& log_destination : log_destinations_) {
    delete log_destination;
    log_destination = nullptr;
  }
  MutexLock l(&sink_mutex_);
  delete sinks_;
  sinks_ = nullptr;
}

namespace {

std::string g_application_fingerprint;

} // namespace

void SetApplicationFingerprint(const std::string& fingerprint) {
  g_application_fingerprint = fingerprint;
}

namespace {

// Directory delimiter; Windows supports both forward slashes and backslashes
#ifdef GLOG_OS_WINDOWS
const char possible_dir_delim[] = {'\\', '/'};
#else
const char possible_dir_delim[] = {'/'};
#endif

string PrettyDuration(int secs) {
  std::stringstream result;
  int mins = secs / 60;
  int hours = mins / 60;
  mins = mins % 60;
  secs = secs % 60;
  result.fill('0');
  result << hours << ':' << setw(2) << mins << ':' << setw(2) << secs;
  return result.str();
}

LogFileObject::LogFileObject(LogSeverity severity, const char* base_filename)
    : base_filename_selected_(base_filename != nullptr),
      base_filename_((base_filename != nullptr) ? base_filename : ""),
      symlink_basename_(glog_internal_namespace_::ProgramInvocationShortName()),
      filename_extension_(),

      severity_(severity),

      rollover_attempt_(kRolloverAttemptFrequency - 1),

      start_time_(WallTime_Now()) {
  assert(severity >= 0);
  assert(severity < NUM_SEVERITIES);
}

LogFileObject::~LogFileObject() {
  MutexLock l(&lock_);
  if (file_ != nullptr) {
    fclose(file_);
    file_ = nullptr;
  }
}

void LogFileObject::SetBasename(const char* basename) {
  MutexLock l(&lock_);
  base_filename_selected_ = true;
  if (base_filename_ != basename) {
    // Get rid of old log file since we are changing names
    if (file_ != nullptr) {
      fclose(file_);
      file_ = nullptr;
      rollover_attempt_ = kRolloverAttemptFrequency-1;
    }
    base_filename_ = basename;
  }
}

void LogFileObject::SetExtension(const char* ext) {
  MutexLock l(&lock_);
  if (filename_extension_ != ext) {
    // Get rid of old log file since we are changing names
    if (file_ != nullptr) {
      fclose(file_);
      file_ = nullptr;
      rollover_attempt_ = kRolloverAttemptFrequency-1;
    }
    filename_extension_ = ext;
  }
}

void LogFileObject::SetSymlinkBasename(const char* symlink_basename) {
  MutexLock l(&lock_);
  symlink_basename_ = symlink_basename;
}

void LogFileObject::Flush() {
  MutexLock l(&lock_);
  FlushUnlocked();
}

void LogFileObject::FlushUnlocked(){
  if (file_ != nullptr) {
    fflush(file_);
    bytes_since_flush_ = 0;
  }
  // Figure out when we are due for another flush.
  const int64 next = (FLAGS_logbufsecs
                      * static_cast<int64>(1000000));  // in usec
  next_flush_time_ = CycleClock_Now() + UsecToCycles(next);
}

bool LogFileObject::CreateLogfile(const string& time_pid_string) {
  string string_filename = base_filename_;
  if (FLAGS_timestamp_in_logfile_name) {
    string_filename += time_pid_string;
  }
  string_filename += filename_extension_;
  const char* filename = string_filename.c_str();
  //only write to files, create if non-existant.
  int flags = O_WRONLY | O_CREAT;
  if (FLAGS_timestamp_in_logfile_name) {
    //demand that the file is unique for our timestamp (fail if it exists).
    flags = flags | O_EXCL;
  }
  int fd = open(filename, flags, static_cast<mode_t>(FLAGS_logfile_mode));
  if (fd == -1) return false;
#ifdef HAVE_FCNTL
  // Mark the file close-on-exec. We don't really care if this fails
  fcntl(fd, F_SETFD, FD_CLOEXEC);

  // Mark the file as exclusive write access to avoid two clients logging to the
  // same file. This applies particularly when !FLAGS_timestamp_in_logfile_name
  // (otherwise open would fail because the O_EXCL flag on similar filename).
  // locks are released on unlock or close() automatically, only after log is
  // released.
  // This will work after a fork as it is not inherited (not stored in the fd).
  // Lock will not be lost because the file is opened with exclusive lock
  // (write) and we will never read from it inside the process.
  // TODO: windows implementation of this (as flock is not available on
  // mingw).
  static struct flock w_lock;

  w_lock.l_type = F_WRLCK;
  w_lock.l_start = 0;
  w_lock.l_whence = SEEK_SET;
  w_lock.l_len = 0;

  int wlock_ret = fcntl(fd, F_SETLK, &w_lock);
  if (wlock_ret == -1) {
      close(fd); //as we are failing already, do not check errors here
      return false;
  }
#endif

  //fdopen in append mode so if the file exists it will fseek to the end
  file_ = fdopen(fd, "a");  // Make a FILE*.
  if (file_ == nullptr) {   // Man, we're screwed!
      close(fd);
      if (FLAGS_timestamp_in_logfile_name) {
      unlink(filename);  // Erase the half-baked evidence: an unusable log file, only if we just created it.
    }
    return false;
  }
#ifdef GLOG_OS_WINDOWS
  // https://github.com/golang/go/issues/27638 - make sure we seek to the end to append
  // empirically replicated with wine over mingw build
  if (!FLAGS_timestamp_in_logfile_name) {
    if (fseek(file_, 0, SEEK_END) != 0) {
      return false;
    }
  }
#endif
  // We try to create a symlink called <program_name>.<severity>,
  // which is easier to use.  (Every time we create a new logfile,
  // we destroy the old symlink and create a new one, so it always
  // points to the latest logfile.)  If it fails, we're sad but it's
  // no error.
  if (!symlink_basename_.empty()) {
    // take directory from filename
    const char* slash = strrchr(filename, PATH_SEPARATOR);
    const string linkname =
      symlink_basename_ + '.' + LogSeverityNames[severity_];
    string linkpath;
    if ( slash ) linkpath = string(filename, static_cast<size_t>(slash-filename+1));  // get dirname
    linkpath += linkname;
    unlink(linkpath.c_str());                    // delete old one if it exists

#if defined(GLOG_OS_WINDOWS)
    // TODO(hamaji): Create lnk file on Windows?
#elif defined(HAVE_UNISTD_H)
    // We must have unistd.h.
    // Make the symlink be relative (in the same dir) so that if the
    // entire log directory gets relocated the link is still valid.
    const char *linkdest = slash ? (slash + 1) : filename;
    if (symlink(linkdest, linkpath.c_str()) != 0) {
      // silently ignore failures
    }

    // Make an additional link to the log file in a place specified by
    // FLAGS_log_link, if indicated
    if (!FLAGS_log_link.empty()) {
      linkpath = FLAGS_log_link + "/" + linkname;
      unlink(linkpath.c_str());                  // delete old one if it exists
      if (symlink(filename, linkpath.c_str()) != 0) {
        // silently ignore failures
      }
    }
#endif
  }

  return true;  // Everything worked
}

void LogFileObject::Write(bool force_flush,
                          time_t timestamp,
                          const char* message,
                          size_t message_len) {
  MutexLock l(&lock_);

  // We don't log if the base_name_ is "" (which means "don't write")
  if (base_filename_selected_ && base_filename_.empty()) {
    return;
  }

  if (file_length_ >> 20U >= MaxLogSize() || PidHasChanged()) {
    if (file_ != nullptr) fclose(file_);
    file_ = nullptr;
    file_length_ = bytes_since_flush_ = dropped_mem_length_ = 0;
    rollover_attempt_ = kRolloverAttemptFrequency - 1;
  }

  // If there's no destination file, make one before outputting
  if (file_ == nullptr) {
    // Try to rollover the log file every 32 log messages.  The only time
    // this could matter would be when we have trouble creating the log
    // file.  If that happens, we'll lose lots of log messages, of course!
    if (++rollover_attempt_ != kRolloverAttemptFrequency) return;
    rollover_attempt_ = 0;

    struct ::tm tm_time;
    if (FLAGS_log_utc_time) {
      gmtime_r(&timestamp, &tm_time);
    } else {
      localtime_r(&timestamp, &tm_time);
    }

    // The logfile's filename will have the date/time & pid in it
    ostringstream time_pid_stream;
    time_pid_stream.fill('0');
    time_pid_stream << 1900+tm_time.tm_year
                    << setw(2) << 1+tm_time.tm_mon
                    << setw(2) << tm_time.tm_mday
                    << '-'
                    << setw(2) << tm_time.tm_hour
                    << setw(2) << tm_time.tm_min
                    << setw(2) << tm_time.tm_sec
                    << '.'
                    << GetMainThreadPid();
    const string& time_pid_string = time_pid_stream.str();

    if (base_filename_selected_) {
      if (!CreateLogfile(time_pid_string)) {
        perror("Could not create log file");
        fprintf(stderr, "COULD NOT CREATE LOGFILE '%s'!\n",
                time_pid_string.c_str());
        return;
      }
    } else {
      // If no base filename for logs of this severity has been set, use a
      // default base filename of
      // "<program name>.<hostname>.<user name>.log.<severity level>.".  So
      // logfiles will have names like
      // webserver.examplehost.root.log.INFO.19990817-150000.4354, where
      // 19990817 is a date (1999 August 17), 150000 is a time (15:00:00),
      // and 4354 is the pid of the logging process.  The date & time reflect
      // when the file was created for output.
      //
      // Where does the file get put?  Successively try the directories
      // "/tmp", and "."
      string stripped_filename(
          glog_internal_namespace_::ProgramInvocationShortName());
      string hostname;
      GetHostName(&hostname);

      string uidname = MyUserName();
      // We should not call CHECK() here because this function can be
      // called after holding on to log_mutex. We don't want to
      // attempt to hold on to the same mutex, and get into a
      // deadlock. Simply use a name like invalid-user.
      if (uidname.empty()) uidname = "invalid-user";

      stripped_filename = stripped_filename+'.'+hostname+'.'
                          +uidname+".log."
                          +LogSeverityNames[severity_]+'.';
      // We're going to (potentially) try to put logs in several different dirs
      const vector<string> & log_dirs = GetLoggingDirectories();

      // Go through the list of dirs, and try to create the log file in each
      // until we succeed or run out of options
      bool success = false;
      for (const auto& log_dir : log_dirs) {
        base_filename_ = log_dir + "/" + stripped_filename;
        if ( CreateLogfile(time_pid_string) ) {
          success = true;
          break;
        }
      }
      // If we never succeeded, we have to give up
      if ( success == false ) {
        perror("Could not create logging file");
        fprintf(stderr, "COULD NOT CREATE A LOGGINGFILE %s!",
                time_pid_string.c_str());
        return;
      }
    }

    // Write a header message into the log file
    if (FLAGS_log_file_header) {
      ostringstream file_header_stream;
      file_header_stream.fill('0');
      file_header_stream << "Log file created at: "
                         << 1900+tm_time.tm_year << '/'
                         << setw(2) << 1+tm_time.tm_mon << '/'
                         << setw(2) << tm_time.tm_mday
                         << ' '
                         << setw(2) << tm_time.tm_hour << ':'
                         << setw(2) << tm_time.tm_min << ':'
                         << setw(2) << tm_time.tm_sec << (FLAGS_log_utc_time ? " UTC\n" : "\n")
                         << "Running on machine: "
                         << LogDestination::hostname() << '\n';

      if(!g_application_fingerprint.empty()) {
        file_header_stream << "Application fingerprint: " << g_application_fingerprint << '\n';
      }
      const char* const date_time_format = FLAGS_log_year_in_prefix
                                               ? "yyyymmdd hh:mm:ss.uuuuuu"
                                               : "mmdd hh:mm:ss.uuuuuu";
      file_header_stream << "Running duration (h:mm:ss): "
                         << PrettyDuration(static_cast<int>(WallTime_Now() - start_time_)) << '\n'
                         << "Log line format: [IWEF]" << date_time_format << " "
                         << "threadid file:line] msg" << '\n';
      const string& file_header_string = file_header_stream.str();

      const size_t header_len = file_header_string.size();
      fwrite(file_header_string.data(), 1, header_len, file_);
      file_length_ += header_len;
      bytes_since_flush_ += header_len;
    }
  }

  // Write to LOG file
  if ( !stop_writing ) {
    // fwrite() doesn't return an error when the disk is full, for
    // messages that are less than 4096 bytes. When the disk is full,
    // it returns the message length for messages that are less than
    // 4096 bytes. fwrite() returns 4096 for message lengths that are
    // greater than 4096, thereby indicating an error.
    errno = 0;
    fwrite(message, 1, message_len, file_);
    if ( FLAGS_stop_logging_if_full_disk &&
         errno == ENOSPC ) {  // disk full, stop writing to disk
      stop_writing = true;  // until the disk is
      return;
    } else {
      file_length_ += message_len;
      bytes_since_flush_ += message_len;
    }
  } else {
    if (CycleClock_Now() >= next_flush_time_) {
      stop_writing = false;  // check to see if disk has free space.
    }
    return;  // no need to flush
  }

  // See important msgs *now*.  Also, flush logs at least every 10^6 chars,
  // or every "FLAGS_logbufsecs" seconds.
  if ( force_flush ||
       (bytes_since_flush_ >= 1000000) ||
       (CycleClock_Now() >= next_flush_time_) ) {
    FlushUnlocked();
#ifdef GLOG_OS_LINUX
    // Only consider files >= 3MiB
    if (FLAGS_drop_log_memory && file_length_ >= (3U << 20U)) {
      // Don't evict the most recent 1-2MiB so as not to impact a tailer
      // of the log file and to avoid page rounding issue on linux < 4.7
      uint32 total_drop_length =
          (file_length_ & ~((1U << 20U) - 1U)) - (1U << 20U);
      uint32 this_drop_length = total_drop_length - dropped_mem_length_;
      if (this_drop_length >= (2U << 20U)) {
        // Only advise when >= 2MiB to drop
# if defined(__ANDROID__) && defined(__ANDROID_API__) && (__ANDROID_API__ < 21)
        // 'posix_fadvise' introduced in API 21:
        // * https://android.googlesource.com/platform/bionic/+/6880f936173081297be0dc12f687d341b86a4cfa/libc/libc.map.txt#732
# else
        posix_fadvise(fileno(file_), static_cast<off_t>(dropped_mem_length_),
                      static_cast<off_t>(this_drop_length),
                      POSIX_FADV_DONTNEED);
# endif
        dropped_mem_length_ = total_drop_length;
      }
    }
#endif

    // Remove old logs
    if (log_cleaner.enabled()) {
      log_cleaner.Run(base_filename_selected_,
                      base_filename_,
                      filename_extension_);
    }
  }
}

LogCleaner::LogCleaner() = default;

void LogCleaner::Enable(unsigned int overdue_days) {
  enabled_ = true;
  overdue_days_ = overdue_days;
}

void LogCleaner::Disable() {
  enabled_ = false;
}

void LogCleaner::UpdateCleanUpTime() {
  const int64 next = (FLAGS_logcleansecs
                      * 1000000);  // in usec
  next_cleanup_time_ = CycleClock_Now() + UsecToCycles(next);
}

void LogCleaner::Run(bool base_filename_selected,
                     const string& base_filename,
                     const string& filename_extension) {
  assert(enabled_);
  assert(!base_filename_selected || !base_filename.empty());

  // avoid scanning logs too frequently
  if (CycleClock_Now() < next_cleanup_time_) {
    return;
  }
  UpdateCleanUpTime();

  vector<string> dirs;

  if (!base_filename_selected) {
    dirs = GetLoggingDirectories();
  } else {
    size_t pos = base_filename.find_last_of(possible_dir_delim, string::npos,
                                            sizeof(possible_dir_delim));
    if (pos != string::npos) {
      string dir = base_filename.substr(0, pos + 1);
      dirs.push_back(dir);
    } else {
      dirs.emplace_back(".");
    }
  }

  for (auto& dir : dirs) {
    vector<string> logs = GetOverdueLogNames(dir, overdue_days_, base_filename,
                                             filename_extension);
    for (auto& log : logs) {
      static_cast<void>(unlink(log.c_str()));
    }
  }
}

vector<string> LogCleaner::GetOverdueLogNames(
    string log_directory, unsigned int days, const string& base_filename,
    const string& filename_extension) const {
  // The names of overdue logs.
  vector<string> overdue_log_names;

  // Try to get all files within log_directory.
  DIR *dir;
  struct dirent *ent;

  if ((dir = opendir(log_directory.c_str()))) {
    while ((ent = readdir(dir))) {
      if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
        continue;
      }

      string filepath = ent->d_name;
      const char* const dir_delim_end =
          possible_dir_delim + sizeof(possible_dir_delim);

      if (!log_directory.empty() &&
          std::find(possible_dir_delim, dir_delim_end,
                    log_directory[log_directory.size() - 1]) != dir_delim_end) {
        filepath = log_directory + filepath;
      }

      if (IsLogFromCurrentProject(filepath, base_filename, filename_extension) &&
          IsLogLastModifiedOver(filepath, days)) {
        overdue_log_names.push_back(filepath);
      }
    }
    closedir(dir);
  }

  return overdue_log_names;
}

bool LogCleaner::IsLogFromCurrentProject(const string& filepath,
                                         const string& base_filename,
                                         const string& filename_extension) const {
  // We should remove duplicated delimiters from `base_filename`, e.g.,
  // before: "/tmp//<base_filename>.<create_time>.<pid>"
  // after:  "/tmp/<base_filename>.<create_time>.<pid>"
  string cleaned_base_filename;

  const char* const dir_delim_end =
      possible_dir_delim + sizeof(possible_dir_delim);

  size_t real_filepath_size = filepath.size();
  for (char c : base_filename) {
    if (cleaned_base_filename.empty()) {
      cleaned_base_filename += c;
    } else if (std::find(possible_dir_delim, dir_delim_end, c) ==
                   dir_delim_end ||
               (!cleaned_base_filename.empty() &&
                c != cleaned_base_filename[cleaned_base_filename.size() - 1])) {
      cleaned_base_filename += c;
    }
  }

  // Return early if the filename doesn't start with `cleaned_base_filename`.
  if (filepath.find(cleaned_base_filename) != 0) {
    return false;
  }

  // Check if in the string `filename_extension` is right next to
  // `cleaned_base_filename` in `filepath` if the user
  // has set a custom filename extension.
  if (!filename_extension.empty()) {
    if (cleaned_base_filename.size() >= real_filepath_size) {
      return false;
    }
    // for origin version, `filename_extension` is middle of the `filepath`.
    string ext = filepath.substr(cleaned_base_filename.size(), filename_extension.size());
    if (ext == filename_extension) {
      cleaned_base_filename += filename_extension;
    }
    else {
      // for new version, `filename_extension` is right of the `filepath`.
      if (filename_extension.size() >= real_filepath_size) {
        return false;
      }
      real_filepath_size = filepath.size() - filename_extension.size();
      if (filepath.substr(real_filepath_size) != filename_extension) {
        return false;
      }
    }
  }

  // The characters after `cleaned_base_filename` should match the format:
  // YYYYMMDD-HHMMSS.pid
  for (size_t i = cleaned_base_filename.size(); i < real_filepath_size; i++) {
    const char& c = filepath[i];

    if (i <= cleaned_base_filename.size() + 7) { // 0 ~ 7 : YYYYMMDD
      if (c < '0' || c > '9') { return false; }

    } else if (i == cleaned_base_filename.size() + 8) { // 8: -
      if (c != '-') { return false; }

    } else if (i <= cleaned_base_filename.size() + 14) { // 9 ~ 14: HHMMSS
      if (c < '0' || c > '9') { return false; }

    } else if (i == cleaned_base_filename.size() + 15) { // 15: .
      if (c != '.') { return false; }

    } else if (i >= cleaned_base_filename.size() + 16) { // 16+: pid
      if (c < '0' || c > '9') { return false; }
    }
  }

  return true;
}

bool LogCleaner::IsLogLastModifiedOver(const string& filepath,
                                       unsigned int days) const {
  // Try to get the last modified time of this file.
  struct stat file_stat;

  if (stat(filepath.c_str(), &file_stat) == 0) {
    const time_t seconds_in_a_day = 60 * 60 * 24;
    time_t last_modified_time = file_stat.st_mtime;
    time_t current_time = time(nullptr);
    return difftime(current_time, last_modified_time) > days * seconds_in_a_day;
  }

  // If failed to get file stat, don't return true!
  return false;
}

}  // namespace

// Static log data space to avoid alloc failures in a LOG(FATAL)
//
// Since multiple threads may call LOG(FATAL), and we want to preserve
// the data from the first call, we allocate two sets of space.  One
// for exclusive use by the first thread, and one for shared use by
// all other threads.
static Mutex fatal_msg_lock;
static CrashReason crash_reason;
static bool fatal_msg_exclusive = true;
static LogMessage::LogMessageData fatal_msg_data_exclusive;
static LogMessage::LogMessageData fatal_msg_data_shared;

#ifdef GLOG_THREAD_LOCAL_STORAGE
// Static thread-local log data space to use, because typically at most one
// LogMessageData object exists (in this case glog makes zero heap memory
// allocations).
static thread_local bool thread_data_available = true;

#if defined(__cpp_lib_byte) && __cpp_lib_byte >= 201603L
// std::aligned_storage is deprecated in C++23
alignas(LogMessage::LogMessageData) static thread_local std::byte
    thread_msg_data[sizeof(LogMessage::LogMessageData)];
#else   // !(defined(__cpp_lib_byte) && __cpp_lib_byte >= 201603L)
static thread_local std::aligned_storage<
    sizeof(LogMessage::LogMessageData),
    alignof(LogMessage::LogMessageData)>::type thread_msg_data;
#endif  // defined(__cpp_lib_byte) && __cpp_lib_byte >= 201603L
#endif  // defined(GLOG_THREAD_LOCAL_STORAGE)

LogMessage::LogMessageData::LogMessageData()
  : stream_(message_text_, LogMessage::kMaxLogMessageLen, 0) {
}

LogMessage::LogMessage(const char* file, int line, LogSeverity severity,
                       int64 ctr, void (LogMessage::*send_method)())
    : allocated_(nullptr) {
  Init(file, line, severity, send_method);
  data_->stream_.set_ctr(ctr);
}

LogMessage::LogMessage(const char* file, int line, const CheckOpString& result)
    : allocated_(nullptr) {
  Init(file, line, GLOG_FATAL, &LogMessage::SendToLog);
  stream() << "Check failed: " << (*result.str_) << " ";
}

LogMessage::LogMessage(const char* file, int line) : allocated_(nullptr) {
  Init(file, line, GLOG_INFO, &LogMessage::SendToLog);
}

LogMessage::LogMessage(const char* file, int line, LogSeverity severity)
    : allocated_(nullptr) {
  Init(file, line, severity, &LogMessage::SendToLog);
}

LogMessage::LogMessage(const char* file, int line, LogSeverity severity,
                       LogSink* sink, bool also_send_to_log)
    : allocated_(nullptr) {
  Init(file, line, severity, also_send_to_log ? &LogMessage::SendToSinkAndLog :
                                                &LogMessage::SendToSink);
  data_->sink_ = sink;  // override Init()'s setting to nullptr
}

LogMessage::LogMessage(const char* file, int line, LogSeverity severity,
                       vector<string>* outvec)
    : allocated_(nullptr) {
  Init(file, line, severity, &LogMessage::SaveOrSendToLog);
  data_->outvec_ = outvec;  // override Init()'s setting to nullptr
}

LogMessage::LogMessage(const char* file, int line, LogSeverity severity,
                       string* message)
    : allocated_(nullptr) {
  Init(file, line, severity, &LogMessage::WriteToStringAndLog);
  data_->message_ = message;  // override Init()'s setting to nullptr
}

void LogMessage::Init(const char* file,
                      int line,
                      LogSeverity severity,
                      void (LogMessage::*send_method)()) {
  allocated_ = nullptr;
  if (severity != GLOG_FATAL || !exit_on_dfatal) {
#ifdef GLOG_THREAD_LOCAL_STORAGE
    // No need for locking, because this is thread local.
    if (thread_data_available) {
      thread_data_available = false;
      data_ = new (&thread_msg_data) LogMessageData;
    } else {
      allocated_ = new LogMessageData();
      data_ = allocated_;
    }
#else // !defined(GLOG_THREAD_LOCAL_STORAGE)
    allocated_ = new LogMessageData();
    data_ = allocated_;
#endif // defined(GLOG_THREAD_LOCAL_STORAGE)
    data_->first_fatal_ = false;
  } else {
    MutexLock l(&fatal_msg_lock);
    if (fatal_msg_exclusive) {
      fatal_msg_exclusive = false;
      data_ = &fatal_msg_data_exclusive;
      data_->first_fatal_ = true;
    } else {
      data_ = &fatal_msg_data_shared;
      data_->first_fatal_ = false;
    }
  }

  data_->preserved_errno_ = errno;
  data_->severity_ = severity;
  data_->line_ = line;
  data_->send_method_ = send_method;
  data_->sink_ = nullptr;
  data_->outvec_ = nullptr;
  WallTime now = WallTime_Now();
  auto timestamp_now = static_cast<time_t>(now);
  logmsgtime_ = LogMessageTime(timestamp_now, now);

  data_->num_chars_to_log_ = 0;
  data_->num_chars_to_syslog_ = 0;
  data_->basename_ = const_basename(file);
  data_->fullname_ = file;
  data_->has_been_flushed_ = false;

  // If specified, prepend a prefix to each line.  For example:
  //    I20201018 160715 f5d4fbb0 logging.cc:1153]
  //    (log level, GMT year, month, date, time, thread_id, file basename, line)
  // We exclude the thread_id for the default thread.
  if (FLAGS_log_prefix && (line != kNoLogPrefix)) {
    std::ios saved_fmt(nullptr);
    saved_fmt.copyfmt(stream());
    stream().fill('0');
    if (custom_prefix_callback == nullptr) {
      stream() << LogSeverityNames[severity][0];
      if (FLAGS_log_year_in_prefix) {
        stream() << setw(4) << 1900 + logmsgtime_.year();
      }
      stream() << setw(2) << 1 + logmsgtime_.month() << setw(2)
               << logmsgtime_.day() << ' ' << setw(2) << logmsgtime_.hour()
               << ':' << setw(2) << logmsgtime_.min() << ':' << setw(2)
               << logmsgtime_.sec() << "." << setw(6) << logmsgtime_.usec()
               << ' ' << setfill(' ') << setw(5)
               << static_cast<unsigned int>(GetTID()) << setfill('0') << ' '
               << data_->basename_ << ':' << data_->line_ << "] ";
    } else {
      custom_prefix_callback(
          stream(),
          LogMessageInfo(LogSeverityNames[severity], data_->basename_,
                         data_->line_, GetTID(), logmsgtime_),
          custom_prefix_callback_data);
      stream() << " ";
    }
      stream().copyfmt(saved_fmt);
  }
  data_->num_prefix_chars_ = data_->stream_.pcount();

  if (!FLAGS_log_backtrace_at.empty()) {
    char fileline[128];
    snprintf(fileline, sizeof(fileline), "%s:%d", data_->basename_, line);
#ifdef HAVE_STACKTRACE
    if (FLAGS_log_backtrace_at == fileline) {
      string stacktrace;
      DumpStackTraceToString(&stacktrace);
      stream() << " (stacktrace:\n" << stacktrace << ") ";
    }
#endif
  }
}

const LogMessageTime& LogMessage::getLogMessageTime() const {
  return logmsgtime_;
}

LogMessage::~LogMessage() {
  Flush();
#ifdef GLOG_THREAD_LOCAL_STORAGE
  if (data_ == static_cast<void*>(&thread_msg_data)) {
    data_->~LogMessageData();
    thread_data_available = true;
  }
  else {
    delete allocated_;
  }
#else // !defined(GLOG_THREAD_LOCAL_STORAGE)
  delete allocated_;
#endif // defined(GLOG_THREAD_LOCAL_STORAGE)
}

int LogMessage::preserved_errno() const {
  return data_->preserved_errno_;
}

ostream& LogMessage::stream() {
  return data_->stream_;
}

// Flush buffered message, called by the destructor, or any other function
// that needs to synchronize the log.
void LogMessage::Flush() {
  if (data_->has_been_flushed_ || data_->severity_ < FLAGS_minloglevel) {
    return;
  }

  data_->num_chars_to_log_ = data_->stream_.pcount();
  data_->num_chars_to_syslog_ =
    data_->num_chars_to_log_ - data_->num_prefix_chars_;

  // Do we need to add a \n to the end of this message?
  bool append_newline =
      (data_->message_text_[data_->num_chars_to_log_-1] != '\n');
  char original_final_char = '\0';

  // If we do need to add a \n, we'll do it by violating the memory of the
  // ostrstream buffer.  This is quick, and we'll make sure to undo our
  // modification before anything else is done with the ostrstream.  It
  // would be preferable not to do things this way, but it seems to be
  // the best way to deal with this.
  if (append_newline) {
    original_final_char = data_->message_text_[data_->num_chars_to_log_];
    data_->message_text_[data_->num_chars_to_log_++] = '\n';
  }
  data_->message_text_[data_->num_chars_to_log_] = '\0';

  // Prevent any subtle race conditions by wrapping a mutex lock around
  // the actual logging action per se.
  {
    MutexLock l(&log_mutex);
    (this->*(data_->send_method_))();
    ++num_messages_[static_cast<int>(data_->severity_)];
  }
  LogDestination::WaitForSinks(data_);

  if (append_newline) {
    // Fix the ostrstream back how it was before we screwed with it.
    // It's 99.44% certain that we don't need to worry about doing this.
    data_->message_text_[data_->num_chars_to_log_-1] = original_final_char;
  }

  // If errno was already set before we enter the logging call, we'll
  // set it back to that value when we return from the logging call.
  // It happens often that we log an error message after a syscall
  // failure, which can potentially set the errno to some other
  // values.  We would like to preserve the original errno.
  if (data_->preserved_errno_ != 0) {
    errno = data_->preserved_errno_;
  }

  // Note that this message is now safely logged.  If we're asked to flush
  // again, as a result of destruction, say, we'll do nothing on future calls.
  data_->has_been_flushed_ = true;
}

// Copy of first FATAL log message so that we can print it out again
// after all the stack traces.  To preserve legacy behavior, we don't
// use fatal_msg_data_exclusive.
static time_t fatal_time;
static char fatal_message[256];

void ReprintFatalMessage() {
  if (fatal_message[0]) {
    const size_t n = strlen(fatal_message);
    if (!FLAGS_logtostderr) {
      // Also write to stderr (don't color to avoid terminal checks)
      WriteToStderr(fatal_message, n);
    }
    LogDestination::LogToAllLogfiles(GLOG_ERROR, fatal_time, fatal_message, n);
  }
}

// L >= log_mutex (callers must hold the log_mutex).
void LogMessage::SendToLog() EXCLUSIVE_LOCKS_REQUIRED(log_mutex) {
  static bool already_warned_before_initgoogle = false;

  log_mutex.AssertHeld();

  RAW_DCHECK(data_->num_chars_to_log_ > 0 &&
             data_->message_text_[data_->num_chars_to_log_-1] == '\n', "");

  // Messages of a given severity get logged to lower severity logs, too

  if (!already_warned_before_initgoogle && !IsGoogleLoggingInitialized()) {
    const char w[] = "WARNING: Logging before InitGoogleLogging() is "
                     "written to STDERR\n";
    WriteToStderr(w, strlen(w));
    already_warned_before_initgoogle = true;
  }

  // global flag: never log to file if set.  Also -- don't log to a
  // file if we haven't parsed the command line flags to get the
  // program name.
  if (FLAGS_logtostderr || FLAGS_logtostdout || !IsGoogleLoggingInitialized()) {
    if (FLAGS_logtostdout) {
      ColoredWriteToStdout(data_->severity_, data_->message_text_,
                           data_->num_chars_to_log_);
    } else {
      ColoredWriteToStderr(data_->severity_, data_->message_text_,
                           data_->num_chars_to_log_);
    }

    // this could be protected by a flag if necessary.
    LogDestination::LogToSinks(data_->severity_,
                               data_->fullname_, data_->basename_,
                               data_->line_, logmsgtime_,
                               data_->message_text_ + data_->num_prefix_chars_,
                               (data_->num_chars_to_log_ -
                                data_->num_prefix_chars_ - 1) );
  } else {
    // log this message to all log files of severity <= severity_
    LogDestination::LogToAllLogfiles(data_->severity_, logmsgtime_.timestamp(),
                                     data_->message_text_,
                                     data_->num_chars_to_log_);

    LogDestination::MaybeLogToStderr(data_->severity_, data_->message_text_,
                                     data_->num_chars_to_log_,
                                     data_->num_prefix_chars_);
    LogDestination::MaybeLogToEmail(data_->severity_, data_->message_text_,
                                    data_->num_chars_to_log_);
    LogDestination::LogToSinks(data_->severity_,
                               data_->fullname_, data_->basename_,
                               data_->line_, logmsgtime_,
                               data_->message_text_ + data_->num_prefix_chars_,
                               (data_->num_chars_to_log_
                                - data_->num_prefix_chars_ - 1) );
    // NOTE: -1 removes trailing \n
  }

  // If we log a FATAL message, flush all the log destinations, then toss
  // a signal for others to catch. We leave the logs in a state that
  // someone else can use them (as long as they flush afterwards)
  if (data_->severity_ == GLOG_FATAL && exit_on_dfatal) {
    if (data_->first_fatal_) {
      // Store crash information so that it is accessible from within signal
      // handlers that may be invoked later.
      RecordCrashReason(&crash_reason);
      SetCrashReason(&crash_reason);

      // Store shortened fatal message for other logs and GWQ status
      const size_t copy = min(data_->num_chars_to_log_,
                                sizeof(fatal_message)-1);
      memcpy(fatal_message, data_->message_text_, copy);
      fatal_message[copy] = '\0';
      fatal_time = logmsgtime_.timestamp();
    }

    if (!FLAGS_logtostderr && !FLAGS_logtostdout) {
      for (auto& log_destination : LogDestination::log_destinations_) {
        if (log_destination) {
          log_destination->logger_->Write(true, 0, "", 0);
        }
      }
    }

    // release the lock that our caller (directly or indirectly)
    // LogMessage::~LogMessage() grabbed so that signal handlers
    // can use the logging facility. Alternately, we could add
    // an entire unsafe logging interface to bypass locking
    // for signal handlers but this seems simpler.
    log_mutex.Unlock();
    LogDestination::WaitForSinks(data_);

    const char* message = "*** Check failure stack trace: ***\n";
    if (write(STDERR_FILENO, message, strlen(message)) < 0) {
      // Ignore errors.
    }
#if defined(__ANDROID__)
    // ANDROID_LOG_FATAL as this message is of FATAL severity.
    __android_log_write(ANDROID_LOG_FATAL,
                        glog_internal_namespace_::ProgramInvocationShortName(),
                        message);
#endif
    Fail();
  }
}

void LogMessage::RecordCrashReason(
    glog_internal_namespace_::CrashReason* reason) {
  reason->filename = fatal_msg_data_exclusive.fullname_;
  reason->line_number = fatal_msg_data_exclusive.line_;
  reason->message = fatal_msg_data_exclusive.message_text_ +
                    fatal_msg_data_exclusive.num_prefix_chars_;
#ifdef HAVE_STACKTRACE
  // Retrieve the stack trace, omitting the logging frames that got us here.
  reason->depth = GetStackTrace(reason->stack, ARRAYSIZE(reason->stack), 4);
#else
  reason->depth = 0;
#endif
}

GLOG_EXPORT logging_fail_func_t g_logging_fail_func =
    reinterpret_cast<logging_fail_func_t>(&abort);

NullStream::NullStream() : LogMessage::LogStream(message_buffer_, 1, 0) {}
NullStream::NullStream(const char* /*file*/, int /*line*/,
                       const CheckOpString& /*result*/)
    : LogMessage::LogStream(message_buffer_, 1, 0) {}
NullStream& NullStream::stream() { return *this; }

NullStreamFatal::~NullStreamFatal() { _exit(EXIT_FAILURE); }

void InstallFailureFunction(logging_fail_func_t fail_func) {
  g_logging_fail_func = fail_func;
}

void LogMessage::Fail() {
  g_logging_fail_func();
}

// L >= log_mutex (callers must hold the log_mutex).
void LogMessage::SendToSink() EXCLUSIVE_LOCKS_REQUIRED(log_mutex) {
  if (data_->sink_ != nullptr) {
    RAW_DCHECK(data_->num_chars_to_log_ > 0 &&
               data_->message_text_[data_->num_chars_to_log_-1] == '\n', "");
    data_->sink_->send(data_->severity_, data_->fullname_, data_->basename_,
                       data_->line_, logmsgtime_,
                       data_->message_text_ + data_->num_prefix_chars_,
                       (data_->num_chars_to_log_ -
                        data_->num_prefix_chars_ - 1) );
  }
}

// L >= log_mutex (callers must hold the log_mutex).
void LogMessage::SendToSinkAndLog() EXCLUSIVE_LOCKS_REQUIRED(log_mutex) {
  SendToSink();
  SendToLog();
}

// L >= log_mutex (callers must hold the log_mutex).
void LogMessage::SaveOrSendToLog() EXCLUSIVE_LOCKS_REQUIRED(log_mutex) {
  if (data_->outvec_ != nullptr) {
    RAW_DCHECK(data_->num_chars_to_log_ > 0 &&
               data_->message_text_[data_->num_chars_to_log_-1] == '\n', "");
    // Omit prefix of message and trailing newline when recording in outvec_.
    const char *start = data_->message_text_ + data_->num_prefix_chars_;
    size_t len = data_->num_chars_to_log_ - data_->num_prefix_chars_ - 1;
    data_->outvec_->push_back(string(start, len));
  } else {
    SendToLog();
  }
}

void LogMessage::WriteToStringAndLog() EXCLUSIVE_LOCKS_REQUIRED(log_mutex) {
  if (data_->message_ != nullptr) {
    RAW_DCHECK(data_->num_chars_to_log_ > 0 &&
               data_->message_text_[data_->num_chars_to_log_-1] == '\n', "");
    // Omit prefix of message and trailing newline when writing to message_.
    const char *start = data_->message_text_ + data_->num_prefix_chars_;
    size_t len = data_->num_chars_to_log_ - data_->num_prefix_chars_ - 1;
    data_->message_->assign(start, len);
  }
  SendToLog();
}

// L >= log_mutex (callers must hold the log_mutex).
void LogMessage::SendToSyslogAndLog() {
#ifdef HAVE_SYSLOG_H
  // Before any calls to syslog(), make a single call to openlog()
  static bool openlog_already_called = false;
  if (!openlog_already_called) {
    openlog(glog_internal_namespace_::ProgramInvocationShortName(),
            LOG_CONS | LOG_NDELAY | LOG_PID,
            LOG_USER);
    openlog_already_called = true;
  }

  // This array maps Google severity levels to syslog levels
  const int SEVERITY_TO_LEVEL[] = { LOG_INFO, LOG_WARNING, LOG_ERR, LOG_EMERG };
  syslog(LOG_USER | SEVERITY_TO_LEVEL[static_cast<int>(data_->severity_)],
         "%.*s", static_cast<int>(data_->num_chars_to_syslog_),
         data_->message_text_ + data_->num_prefix_chars_);
  SendToLog();
#else
  LOG(ERROR) << "No syslog support: message=" << data_->message_text_;
#endif
}

base::Logger* base::GetLogger(LogSeverity severity) {
  MutexLock l(&log_mutex);
  return LogDestination::log_destination(severity)->GetLoggerImpl();
}

void base::SetLogger(LogSeverity severity, base::Logger* logger) {
  MutexLock l(&log_mutex);
  LogDestination::log_destination(severity)->SetLoggerImpl(logger);
}

// L < log_mutex.  Acquires and releases mutex_.
int64 LogMessage::num_messages(int severity) {
  MutexLock l(&log_mutex);
  return num_messages_[severity];
}

// Output the COUNTER value. This is only valid if ostream is a
// LogStream.
ostream& operator<<(ostream &os, const PRIVATE_Counter&) {
#ifdef DISABLE_RTTI
  LogMessage::LogStream *log = static_cast<LogMessage::LogStream*>(&os);
#else
  auto* log = dynamic_cast<LogMessage::LogStream*>(&os);
#endif
  CHECK(log && log == log->self())
      << "You must not use COUNTER with non-glog ostream";
  os << log->ctr();
  return os;
}

ErrnoLogMessage::ErrnoLogMessage(const char* file, int line,
                                 LogSeverity severity, int64 ctr,
                                 void (LogMessage::*send_method)())
    : LogMessage(file, line, severity, ctr, send_method) {}

ErrnoLogMessage::~ErrnoLogMessage() {
  // Don't access errno directly because it may have been altered
  // while streaming the message.
  stream() << ": " << StrError(preserved_errno()) << " ["
           << preserved_errno() << "]";
}

void FlushLogFiles(LogSeverity min_severity) {
  LogDestination::FlushLogFiles(min_severity);
}

void FlushLogFilesUnsafe(LogSeverity min_severity) {
  LogDestination::FlushLogFilesUnsafe(min_severity);
}

void SetLogDestination(LogSeverity severity, const char* base_filename) {
  LogDestination::SetLogDestination(severity, base_filename);
}

void SetLogSymlink(LogSeverity severity, const char* symlink_basename) {
  LogDestination::SetLogSymlink(severity, symlink_basename);
}

LogSink::~LogSink() = default;

void LogSink::send(LogSeverity severity, const char* full_filename,
                   const char* base_filename, int line,
                   const LogMessageTime& time, const char* message,
                   size_t message_len) {
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif  // __GNUC__
  send(severity, full_filename, base_filename, line, &time.tm(), message,
       message_len);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif  // __GNUC__
}

void LogSink::send(LogSeverity severity, const char* full_filename,
                   const char* base_filename, int line, const std::tm* t,
                   const char* message, size_t message_len) {
  (void)severity;
  (void)full_filename;
  (void)base_filename;
  (void)line;
  (void)t;
  (void)message;
  (void)message_len;
}

void LogSink::WaitTillSent() {
  // noop default
}

string LogSink::ToString(LogSeverity severity, const char* file, int line,
                         const LogMessageTime& logmsgtime, const char* message,
                         size_t message_len) {
  ostringstream stream;
  stream.fill('0');

  stream << LogSeverityNames[severity][0];
  if (FLAGS_log_year_in_prefix) {
    stream << setw(4) << 1900 + logmsgtime.year();
  }
  stream << setw(2) << 1 + logmsgtime.month()
         << setw(2) << logmsgtime.day()
         << ' '
         << setw(2) << logmsgtime.hour() << ':'
         << setw(2) << logmsgtime.min() << ':'
         << setw(2) << logmsgtime.sec() << '.'
         << setw(6) << logmsgtime.usec()
         << ' '
         << setfill(' ') << setw(5) << GetTID() << setfill('0')
         << ' '
         << file << ':' << line << "] ";

  // A call to `write' is enclosed in parenthneses to prevent possible macro
  // expansion.  On Windows, `write' could be a macro defined for portability.
  (stream.write)(message, static_cast<std::streamsize>(message_len));
  return stream.str();
}

void AddLogSink(LogSink *destination) {
  LogDestination::AddLogSink(destination);
}

void RemoveLogSink(LogSink *destination) {
  LogDestination::RemoveLogSink(destination);
}

void SetLogFilenameExtension(const char* ext) {
  LogDestination::SetLogFilenameExtension(ext);
}

void SetStderrLogging(LogSeverity min_severity) {
  LogDestination::SetStderrLogging(min_severity);
}

void SetEmailLogging(LogSeverity min_severity, const char* addresses) {
  LogDestination::SetEmailLogging(min_severity, addresses);
}

void LogToStderr() {
  LogDestination::LogToStderr();
}

namespace base {
namespace internal {

bool GetExitOnDFatal();
bool GetExitOnDFatal() {
  MutexLock l(&log_mutex);
  return exit_on_dfatal;
}

// Determines whether we exit the program for a LOG(DFATAL) message in
// debug mode.  It does this by skipping the call to Fail/FailQuietly.
// This is intended for testing only.
//
// This can have some effects on LOG(FATAL) as well.  Failure messages
// are always allocated (rather than sharing a buffer), the crash
// reason is not recorded, the "gwq" status message is not updated,
// and the stack trace is not recorded.  The LOG(FATAL) *will* still
// exit the program.  Since this function is used only in testing,
// these differences are acceptable.
void SetExitOnDFatal(bool value);
void SetExitOnDFatal(bool value) {
  MutexLock l(&log_mutex);
  exit_on_dfatal = value;
}

}  // namespace internal
}  // namespace base

#ifndef GLOG_OS_EMSCRIPTEN
// Shell-escaping as we need to shell out ot /bin/mail.
static const char kDontNeedShellEscapeChars[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789+-_.=/:,@";

static string ShellEscape(const string& src) {
  string result;
  if (!src.empty() &&  // empty string needs quotes
      src.find_first_not_of(kDontNeedShellEscapeChars) == string::npos) {
    // only contains chars that don't need quotes; it's fine
    result.assign(src);
  } else if (src.find_first_of('\'') == string::npos) {
    // no single quotes; just wrap it in single quotes
    result.assign("'");
    result.append(src);
    result.append("'");
  } else {
    // needs double quote escaping
    result.assign("\"");
    for (size_t i = 0; i < src.size(); ++i) {
      switch (src[i]) {
        case '\\':
        case '$':
        case '"':
        case '`':
          result.append("\\");
      }
      result.append(src, i, 1);
    }
    result.append("\"");
  }
  return result;
}

// Trim whitespace from both ends of the provided string.
static inline void trim(std::string &s) {
  const auto toRemove = [](char ch) { return std::isspace(ch) == 0; };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), toRemove));
  s.erase(std::find_if(s.rbegin(), s.rend(), toRemove).base(), s.end());
}
#endif

// use_logging controls whether the logging functions LOG/VLOG are used
// to log errors.  It should be set to false when the caller holds the
// log_mutex.
static bool SendEmailInternal(const char*dest, const char *subject,
                              const char*body, bool use_logging) {
#ifndef GLOG_OS_EMSCRIPTEN
  if (dest && *dest) {
    // Split the comma-separated list of email addresses, validate each one and
    // build a sanitized new comma-separated string without whitespace.
    std::istringstream ss(dest);
    std::ostringstream sanitized_dests;
    std::string s;
    while (std::getline(ss, s, ',')) {
      trim(s);
      if (s.empty()) {
        continue;
      }
      // We validate the provided email addresses using the same regular
      // expression that HTML5 uses[1], except that we require the address to
      // start with an alpha-numeric character. This is because we don't want to
      // allow email addresses that start with a special character, such as a
      // pipe or dash, which could be misunderstood as a command-line flag by
      // certain versions of `mail` that are vulnerable to command injection.[2]
      // [1] https://html.spec.whatwg.org/multipage/input.html#valid-e-mail-address
      // [2] e.g. https://nvd.nist.gov/vuln/detail/CVE-2004-2771
      if (!std::regex_match(
              s,
              std::regex("^[a-zA-Z0-9]"
                         "[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]*@[a-zA-Z0-9]"
                         "(?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\\.[a-zA-Z0-9]"
                         "(?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"))) {
        if (use_logging) {
          VLOG(1) << "Invalid destination email address:" << s;
        } else {
          fprintf(stderr, "Invalid destination email address: %s\n",
                  s.c_str());
        }
        return false;
      }
      if (!sanitized_dests.str().empty()) {
        sanitized_dests << ",";
      }
      sanitized_dests << s;
    }
    // Avoid dangling reference
    const std::string& tmp = sanitized_dests.str();
    dest = tmp.c_str();

    if ( use_logging ) {
      VLOG(1) << "Trying to send TITLE:" << subject
              << " BODY:" << body << " to " << dest;
    } else {
      fprintf(stderr, "Trying to send TITLE: %s BODY: %s to %s\n",
              subject, body, dest);
    }

    string logmailer = FLAGS_logmailer;
    if (logmailer.empty()) {
        logmailer = "/bin/mail";
    }
    string cmd =
        logmailer + " -s" +
        ShellEscape(subject) + " " + ShellEscape(dest);
    if (use_logging) {
        VLOG(4) << "Mailing command: " << cmd;
    }

    FILE* pipe = popen(cmd.c_str(), "w");
    if (pipe != nullptr) {
      // Add the body if we have one
      if (body) {
        fwrite(body, sizeof(char), strlen(body), pipe);
      }
      bool ok = pclose(pipe) != -1;
      if ( !ok ) {
        if ( use_logging ) {
          LOG(ERROR) << "Problems sending mail to " << dest << ": "
                     << StrError(errno);
        } else {
          fprintf(stderr, "Problems sending mail to %s: %s\n",
                  dest, StrError(errno).c_str());
        }
      }
      return ok;
    } else {
      if ( use_logging ) {
        LOG(ERROR) << "Unable to send mail to " << dest;
      } else {
        fprintf(stderr, "Unable to send mail to %s\n", dest);
      }
    }
  }
#else
  (void)dest;
  (void)subject;
  (void)body;
  (void)use_logging;
  LOG(WARNING) << "Email support not available; not sending message";
#endif
  return false;
}

bool SendEmail(const char*dest, const char *subject, const char*body){
  return SendEmailInternal(dest, subject, body, true);
}

static void GetTempDirectories(vector<string>* list) {
  list->clear();
#ifdef GLOG_OS_WINDOWS
  // On windows we'll try to find a directory in this order:
  //   C:/Documents & Settings/whomever/TEMP (or whatever GetTempPath() is)
  //   C:/TMP/
  //   C:/TEMP/
  //   C:/WINDOWS/ or C:/WINNT/
  //   .
  char tmp[MAX_PATH];
  if (GetTempPathA(MAX_PATH, tmp))
    list->push_back(tmp);
  list->push_back("C:\\tmp\\");
  list->push_back("C:\\temp\\");
#else
  // Directories, in order of preference. If we find a dir that
  // exists, we stop adding other less-preferred dirs
  const char * candidates[] = {
    // Non-null only during unittest/regtest
    getenv("TEST_TMPDIR"),

    // Explicitly-supplied temp dirs
    getenv("TMPDIR"), getenv("TMP"),

    // If all else fails
    "/tmp",
  };

  for (auto d : candidates) {
    if (!d) continue;  // Empty env var

    // Make sure we don't surprise anyone who's expecting a '/'
    string dstr = d;
    if (dstr[dstr.size() - 1] != '/') {
      dstr += "/";
    }
    list->push_back(dstr);

    struct stat statbuf;
    if (!stat(d, &statbuf) && S_ISDIR(statbuf.st_mode)) {
      // We found a dir that exists - we're done.
      return;
    }
  }

#endif
}

static vector<string>* logging_directories_list;

const vector<string>& GetLoggingDirectories() {
  // Not strictly thread-safe but we're called early in InitGoogle().
  if (logging_directories_list == nullptr) {
    logging_directories_list = new vector<string>;

    if ( !FLAGS_log_dir.empty() ) {
      // Ensure the specified path ends with a directory delimiter.
      if (std::find(std::begin(possible_dir_delim), std::end(possible_dir_delim),
            FLAGS_log_dir.back()) == std::end(possible_dir_delim)) {
        logging_directories_list->push_back(FLAGS_log_dir + "/");
      } else {
        logging_directories_list->push_back(FLAGS_log_dir);
      }
    } else {
      GetTempDirectories(logging_directories_list);
#ifdef GLOG_OS_WINDOWS
      char tmp[MAX_PATH];
      if (GetWindowsDirectoryA(tmp, MAX_PATH))
        logging_directories_list->push_back(tmp);
      logging_directories_list->push_back(".\\");
#else
      logging_directories_list->push_back("./");
#endif
    }
  }
  return *logging_directories_list;
}

void TestOnly_ClearLoggingDirectoriesList() {
  fprintf(stderr, "TestOnly_ClearLoggingDirectoriesList should only be "
          "called from test code.\n");
  delete logging_directories_list;
  logging_directories_list = nullptr;
}

void GetExistingTempDirectories(vector<string>* list) {
  GetTempDirectories(list);
  auto i_dir = list->begin();
  while( i_dir != list->end() ) {
    // zero arg to access means test for existence; no constant
    // defined on windows
    if ( access(i_dir->c_str(), 0) ) {
      i_dir = list->erase(i_dir);
    } else {
      ++i_dir;
    }
  }
}

void TruncateLogFile(const char *path, uint64 limit, uint64 keep) {
#if defined(HAVE_UNISTD_H) || defined(HAVE__CHSIZE_S)
  struct stat statbuf;
  const int kCopyBlockSize = 8 << 10;
  char copybuf[kCopyBlockSize];
  off_t read_offset, write_offset;
  // Don't follow symlinks unless they're our own fd symlinks in /proc
  int flags = O_RDWR;
  // TODO(hamaji): Support other environments.
#ifdef GLOG_OS_LINUX
  const char *procfd_prefix = "/proc/self/fd/";
  if (strncmp(procfd_prefix, path, strlen(procfd_prefix))) flags |= O_NOFOLLOW;
#endif

  int fd = open(path, flags);
  if (fd == -1) {
    if (errno == EFBIG) {
      // The log file in question has got too big for us to open. The
      // real fix for this would be to compile logging.cc (or probably
      // all of base/...) with -D_FILE_OFFSET_BITS=64 but that's
      // rather scary.
      // Instead just truncate the file to something we can manage
#ifdef HAVE__CHSIZE_S
      if (_chsize_s(fd, 0) != 0) {
#else
      if (truncate(path, 0) == -1) {
#endif
        PLOG(ERROR) << "Unable to truncate " << path;
      } else {
        LOG(ERROR) << "Truncated " << path << " due to EFBIG error";
      }
    } else {
      PLOG(ERROR) << "Unable to open " << path;
    }
    return;
  }

  if (fstat(fd, &statbuf) == -1) {
    PLOG(ERROR) << "Unable to fstat()";
    goto out_close_fd;
  }

  // See if the path refers to a regular file bigger than the
  // specified limit
  if (!S_ISREG(statbuf.st_mode)) goto out_close_fd;
  if (statbuf.st_size <= static_cast<off_t>(limit))  goto out_close_fd;
  if (statbuf.st_size <= static_cast<off_t>(keep)) goto out_close_fd;

  // This log file is too large - we need to truncate it
  LOG(INFO) << "Truncating " << path << " to " << keep << " bytes";

  // Copy the last "keep" bytes of the file to the beginning of the file
  read_offset = statbuf.st_size - static_cast<off_t>(keep);
  write_offset = 0;
  ssize_t bytesin, bytesout;
  while ((bytesin = pread(fd, copybuf, sizeof(copybuf), read_offset)) > 0) {
    bytesout = pwrite(fd, copybuf, static_cast<size_t>(bytesin), write_offset);
    if (bytesout == -1) {
      PLOG(ERROR) << "Unable to write to " << path;
      break;
    } else if (bytesout != bytesin) {
      LOG(ERROR) << "Expected to write " << bytesin << ", wrote " << bytesout;
    }
    read_offset += bytesin;
    write_offset += bytesout;
  }
  if (bytesin == -1) PLOG(ERROR) << "Unable to read from " << path;

  // Truncate the remainder of the file. If someone else writes to the
  // end of the file after our last read() above, we lose their latest
  // data. Too bad ...
#ifdef HAVE__CHSIZE_S
  if (_chsize_s(fd, write_offset) != 0) {
#else
  if (ftruncate(fd, write_offset) == -1) {
#endif
    PLOG(ERROR) << "Unable to truncate " << path;
  }

 out_close_fd:
  close(fd);
#else
  LOG(ERROR) << "No log truncation support.";
#endif
 }

void TruncateStdoutStderr() {
#ifdef HAVE_UNISTD_H
  uint64 limit = MaxLogSize() << 20U;
  uint64 keep = 1U << 20U;
  TruncateLogFile("/proc/self/fd/1", limit, keep);
  TruncateLogFile("/proc/self/fd/2", limit, keep);
#else
  LOG(ERROR) << "No log truncation support.";
#endif
}


// Helper functions for string comparisons.
#define DEFINE_CHECK_STROP_IMPL(name, func, expected)                         \
  string* Check##func##expected##Impl(const char* s1, const char* s2,         \
                                      const char* names) {                    \
    bool equal = s1 == s2 || (s1 && s2 && !func(s1, s2));                     \
    if (equal == expected)                                                    \
      return nullptr;                                                         \
    else {                                                                    \
      ostringstream ss;                                                       \
      if (!s1) s1 = "";                                                       \
      if (!s2) s2 = "";                                                       \
      ss << #name " failed: " << names << " (" << s1 << " vs. " << s2 << ")"; \
      return new string(ss.str());                                            \
    }                                                                         \
  }
DEFINE_CHECK_STROP_IMPL(CHECK_STREQ, strcmp, true)
DEFINE_CHECK_STROP_IMPL(CHECK_STRNE, strcmp, false)
DEFINE_CHECK_STROP_IMPL(CHECK_STRCASEEQ, strcasecmp, true)
DEFINE_CHECK_STROP_IMPL(CHECK_STRCASENE, strcasecmp, false)
#undef DEFINE_CHECK_STROP_IMPL

int posix_strerror_r(int err, char *buf, size_t len) {
  // Sanity check input parameters
  if (buf == nullptr || len <= 0) {
    errno = EINVAL;
    return -1;
  }

  // Reset buf and errno, and try calling whatever version of strerror_r()
  // is implemented by glibc
  buf[0] = '\000';
  int old_errno = errno;
  errno = 0;
  char *rc = reinterpret_cast<char *>(strerror_r(err, buf, len));

  // Both versions set errno on failure
  if (errno) {
    // Should already be there, but better safe than sorry
    buf[0]     = '\000';
    return -1;
  }
  errno = old_errno;

  // POSIX is vague about whether the string will be terminated, although
  // is indirectly implies that typically ERANGE will be returned, instead
  // of truncating the string. This is different from the GNU implementation.
  // We play it safe by always terminating the string explicitly.
  buf[len-1] = '\000';

  // If the function succeeded, we can use its exit code to determine the
  // semantics implemented by glibc
  if (!rc) {
    return 0;
  } else {
    // GNU semantics detected
    if (rc == buf) {
      return 0;
    } else {
      buf[0] = '\000';
#if defined(GLOG_OS_MACOSX) || defined(GLOG_OS_FREEBSD) || defined(GLOG_OS_OPENBSD)
      if (reinterpret_cast<intptr_t>(rc) < sys_nerr) {
        // This means an error on MacOSX or FreeBSD.
        return -1;
      }
#endif
      strncat(buf, rc, len-1);
      return 0;
    }
  }
}

string StrError(int err) {
  char buf[100];
  int rc = posix_strerror_r(err, buf, sizeof(buf));
  if ((rc < 0) || (buf[0] == '\000')) {
    snprintf(buf, sizeof(buf), "Error number %d", err);
  }
  return buf;
}

LogMessageFatal::LogMessageFatal(const char* file, int line) :
    LogMessage(file, line, GLOG_FATAL) {}

LogMessageFatal::LogMessageFatal(const char* file, int line,
                                 const CheckOpString& result) :
    LogMessage(file, line, result) {}

LogMessageFatal::~LogMessageFatal() {
    Flush();
    LogMessage::Fail();
}

namespace base {

CheckOpMessageBuilder::CheckOpMessageBuilder(const char *exprtext)
    : stream_(new ostringstream) {
  *stream_ << exprtext << " (";
}

CheckOpMessageBuilder::~CheckOpMessageBuilder() {
  delete stream_;
}

ostream* CheckOpMessageBuilder::ForVar2() {
  *stream_ << " vs. ";
  return stream_;
}

string* CheckOpMessageBuilder::NewString() {
  *stream_ << ")";
  return new string(stream_->str());
}

}  // namespace base

template <>
void MakeCheckOpValueString(std::ostream* os, const char& v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "char value " << static_cast<short>(v);
  }
}

template <>
void MakeCheckOpValueString(std::ostream* os, const signed char& v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "signed char value " << static_cast<short>(v);
  }
}

template <>
void MakeCheckOpValueString(std::ostream* os, const unsigned char& v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "unsigned char value " << static_cast<unsigned short>(v);
  }
}

template <>
void MakeCheckOpValueString(std::ostream* os, const std::nullptr_t& /*v*/) {
  (*os) << "nullptr";
}

void InitGoogleLogging(const char* argv0) {
  glog_internal_namespace_::InitGoogleLoggingUtilities(argv0);
}

void InitGoogleLogging(const char* argv0,
                       CustomPrefixCallback prefix_callback,
                       void* prefix_callback_data) {
  custom_prefix_callback = prefix_callback;
  custom_prefix_callback_data = prefix_callback_data;
  InitGoogleLogging(argv0);
}

void ShutdownGoogleLogging() {
  glog_internal_namespace_::ShutdownGoogleLoggingUtilities();
  LogDestination::DeleteLogDestinations();
  delete logging_directories_list;
  logging_directories_list = nullptr;
}

void EnableLogCleaner(unsigned int overdue_days) {
  log_cleaner.Enable(overdue_days);
}

void DisableLogCleaner() {
  log_cleaner.Disable();
}

LogMessageTime::LogMessageTime()
    : time_struct_(), timestamp_(0), usecs_(0), gmtoffset_(0) {}

LogMessageTime::LogMessageTime(std::tm t) {
  std::time_t timestamp = std::mktime(&t);
  init(t, timestamp, 0);
}

LogMessageTime::LogMessageTime(std::time_t timestamp, WallTime now) {
  std::tm t;
  if (FLAGS_log_utc_time) {
    gmtime_r(&timestamp, &t);
  } else {
    localtime_r(&timestamp, &t);
  }
  init(t, timestamp, now);
}

void LogMessageTime::init(const std::tm& t, std::time_t timestamp,
                          WallTime now) {
  time_struct_ = t;
  timestamp_ = timestamp;
  usecs_ = static_cast<int32>((now - timestamp) * 1000000);

  CalcGmtOffset();
}

void LogMessageTime::CalcGmtOffset() {
  std::tm gmt_struct;
  int isDst = 0;
  if ( FLAGS_log_utc_time ) {
    localtime_r(&timestamp_, &gmt_struct);
    isDst = gmt_struct.tm_isdst;
    gmt_struct = time_struct_;
  } else {
    isDst = time_struct_.tm_isdst;
    gmtime_r(&timestamp_, &gmt_struct);
  }

  time_t gmt_sec = mktime(&gmt_struct);
  const long hour_secs = 3600;
  // If the Daylight Saving Time(isDst) is active subtract an hour from the current timestamp.
  gmtoffset_ = static_cast<long int>(timestamp_ - gmt_sec + (isDst ? hour_secs : 0) ) ;
}

_END_GOOGLE_NAMESPACE_
