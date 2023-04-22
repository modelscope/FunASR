/**
 * Copyright      2022  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gtest/gtest.h"
#include "kaldi-native-fbank/csrc/log.h"

namespace knf {

#if KNF_ENABLE_CHECK

TEST(Log, TestLog) {
  KNF_LOG(TRACE) << "this is a trace message";
  KNF_LOG(DEBUG) << "this is a debug message";
  KNF_LOG(INFO) << "this is an info message";
  KNF_LOG(WARNING) << "this is a warning message";
  KNF_LOG(ERROR) << "this is an error message";

  ASSERT_THROW(KNF_LOG(FATAL) << "This will crash the program",
               std::runtime_error);

  // For debug build

  KNF_DLOG(TRACE) << "this is a trace message for debug build";
  KNF_DLOG(DEBUG) << "this is a trace message for debug build";
  KNF_DLOG(INFO) << "this is a trace message for debug build";
  KNF_DLOG(ERROR) << "this is an error message for debug build";
  KNF_DLOG(WARNING) << "this is a trace message for debug build";

#if !defined(NDEBUG)
  ASSERT_THROW(KNF_DLOG(FATAL) << "this is a trace message for debug build",
               std::runtime_error);
#endif
}

TEST(Log, TestCheck) {
  KNF_CHECK_EQ(1, 1) << "ok";
  KNF_CHECK_LE(1, 3) << "ok";
  KNF_CHECK_LT(1, 2) << "ok";
  KNF_CHECK_GT(2, 1) << "ok";
  KNF_CHECK_GE(2, 1) << "ok";

  ASSERT_THROW(KNF_CHECK_EQ(2, 1) << "bad things happened", std::runtime_error);

  // for debug build
  KNF_DCHECK_EQ(1, 1) << "ok";
  KNF_DCHECK_LE(1, 3) << "ok";
  KNF_DCHECK_LT(1, 2) << "ok";
  KNF_DCHECK_GT(2, 1) << "ok";
  KNF_DCHECK_GE(2, 1) << "ok";

#if !defined(NDEBUG)
  ASSERT_THROW(KNF_CHECK_EQ(2, 1) << "bad things happened", std::runtime_error);
#endif
}

#endif

}  // namespace knf
