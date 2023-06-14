// itf/options-itf.h

// Copyright 2013  Tanel Alumae, Tallinn University of Technology

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_ITF_OPTIONS_ITF_H_
#define KALDI_ITF_OPTIONS_ITF_H_ 1
#include "base/kaldi-common.h"

namespace kaldi {

class OptionsItf {
 public:
  
  virtual void Register(const std::string &name,
                bool *ptr, const std::string &doc) = 0; 
  virtual void Register(const std::string &name,
                int32 *ptr, const std::string &doc) = 0; 
  virtual void Register(const std::string &name,
                uint32 *ptr, const std::string &doc) = 0; 
  virtual void Register(const std::string &name,
                float *ptr, const std::string &doc) = 0; 
  virtual void Register(const std::string &name,
                double *ptr, const std::string &doc) = 0; 
  virtual void Register(const std::string &name,
                std::string *ptr, const std::string &doc) = 0; 
  
  virtual ~OptionsItf() {}
};

}  // namespace Kaldi

#endif  // KALDI_ITF_OPTIONS_ITF_H_


