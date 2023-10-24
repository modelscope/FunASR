// util/kaldi-cygwin-io-inl.h

// Copyright 2015 Smart Action Company LLC (author: Kirill Katsnelson)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.
#ifndef KALDI_UTIL_KALDI_CYGWIN_IO_INL_H_
#define KALDI_UTIL_KALDI_CYGWIN_IO_INL_H_

#ifndef _MSC_VER
#error This is a Windows-compatibility file. Something went wery wrong.
#endif

#include <string>

// This file is included only into kaldi-io.cc, and only if
// KALDI_CYGWIN_COMPAT is enabled.
//
// The routines map unix-ey paths passed to Windows programs from shell
// scripts in egs. Since shell scripts run under cygwin, they use cygwin's
// own mount table and a mapping to the file system. It is quite possible to
// create quite an intricate mapping that only own cygwin API would be able
// to untangle. Unfortunately, the API to map between filenames is not
// available to non-cygwin programs. Running cygpath for every file operation
// would as well be cumbersome. So this is only a simplistic path resolution,
// assuming that the default cygwin prefix /cygdrive is used, and that all
// resolved unix-style full paths end up prefixed with /cygdrive. This is
// quite a sensible approach. We'll also try to map /dev/null and /tmp/**,
// die on all other /dev/** and warn about all other rooted paths.

namespace kaldi {

static bool prefixp(const std::string& pfx, const std::string& str) {
  return pfx.length() <= str.length() &&
    std::equal(pfx.begin(), pfx.end(), str.begin());
}

static std::string cygprefix("/cygdrive/");

static std::string MapCygwinPathNoTmp(const std::string &filename) {
  // UNC(?), relative, native Windows and empty paths are ok already.
  if (prefixp("//", filename) || !prefixp("/", filename))
    return filename;

  // /dev/...
  if (filename == "/dev/null")
    return "\\\\.\\nul";
  if (prefixp("/dev/", filename)) {
      KALDI_ERR << "Unable to resolve path '" << filename
                << "' - only have /dev/null here.";
      return "\\\\.\\invalid";
  }

  // /cygdrive/?[/....]
  int preflen = cygprefix.size();
  if (prefixp(cygprefix, filename)
      && filename.size() >= preflen + 1 && isalpha(filename[preflen])
      && (filename.size() == preflen + 1 || filename[preflen + 1] == '/')) {
    return std::string() + filename[preflen] + ':' +
       (filename.size() > preflen + 1 ? filename.substr(preflen + 1) : "/");
  }

  KALDI_WARN << "Unable to resolve path '" << filename
             << "' - cannot map unix prefix. "
             << "Will go on, but breakage will likely ensue.";
  return filename;
}

// extern for unit testing.
std::string MapCygwinPath(const std::string &filename) {
  // /tmp[/....]
  if (filename != "/tmp" && !prefixp("/tmp/", filename)) {
    return MapCygwinPathNoTmp(filename);
  }
  char *tmpdir = std::getenv("TMP");
  if (tmpdir == nullptr)
    tmpdir = std::getenv("TEMP");
  if (tmpdir == nullptr) {
    KALDI_ERR << "Unable to resolve path '" << filename
              << "' - unable to find temporary directory. Set TMP.";
    return filename;
  }
  // Map the value of tmpdir again, as cygwin environment actually may contain
  // unix-style paths.
  return MapCygwinPathNoTmp(std::string(tmpdir) + filename.substr(4));
}

// A popen implementation that passes the command line through cygwin
// bash.exe. This is necessary since some piped commands are cygwin links
// (e. g. fgrep is a soft link to grep), and some are #!-files, such as
// gunzip which is a shell script that invokes gzip, or kaldi's own run.pl
// which is a perl script.
//
// _popen uses cmd.exe or whatever shell is specified via the COMSPEC
// variable. Unfortunately, it adds a hardcoded " /c " to it, so we cannot
// just substitute the environment variable COMSPEC to point to bash.exe.
// Instead, quote the command and pass it to bash via its -c switch.
static FILE *CygwinCompatPopen(const char* command, const char* mode) {
  // To speed up command launch marginally, optionally accept full path
  // to bash.exe. This will not work if the path contains spaces, but
  // no sane person would install cygwin into a space-ridden path.
  const char* bash_exe = std::getenv("BASH_EXE");
  std::string qcmd(bash_exe != nullptr ? bash_exe : "bash.exe");
  qcmd += " -c \"";
  for (; *command; ++command) {
    if (*command == '\"')
      qcmd += '\"';
    qcmd += *command;
  }
  qcmd += '\"';

  return _popen(qcmd.c_str(), mode);
}

}  // namespace kaldi

#endif  // KALDI_UTIL_KALDI_CYGWIN_IO_INL_H_
