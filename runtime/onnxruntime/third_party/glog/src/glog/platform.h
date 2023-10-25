// Copyright (c) 2008, Google Inc.
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
//
// Author: Shinichiro Hamaji
//
// Detect supported platforms.

#ifndef GLOG_PLATFORM_H
#define GLOG_PLATFORM_H

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#define GLOG_OS_WINDOWS
#elif defined(__CYGWIN__) || defined(__CYGWIN32__)
#define GLOG_OS_CYGWIN
#elif defined(linux) || defined(__linux) || defined(__linux__)
#ifndef GLOG_OS_LINUX
#define GLOG_OS_LINUX
#endif
#elif defined(macintosh) || defined(__APPLE__) || defined(__APPLE_CC__)
#define GLOG_OS_MACOSX
#elif defined(__FreeBSD__)
#define GLOG_OS_FREEBSD
#elif defined(__NetBSD__)
#define GLOG_OS_NETBSD
#elif defined(__OpenBSD__)
#define GLOG_OS_OPENBSD
#elif defined(__EMSCRIPTEN__)
#define GLOG_OS_EMSCRIPTEN
#else
// TODO(hamaji): Add other platforms.
#error Platform not supported by glog. Please consider to contribute platform information by submitting a pull request on Github.
#endif

#endif // GLOG_PLATFORM_H
