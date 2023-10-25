// itf/optimizable-itf.h

// Copyright 2009-2011  Go Vivace Inc.;  Microsoft Corporation;  Georg Stemmer

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
#ifndef KALDI_ITF_OPTIMIZABLE_ITF_H_
#define KALDI_ITF_OPTIMIZABLE_ITF_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"

namespace kaldi {
/// @ingroup Interfaces
/// @{

/// OptimizableInterface provides
/// a virtual class for optimizable objects.
/// E.g. a class that computed a likelihood function and
/// its gradient using some parameter
/// that has to be optimized on data
/// could inherit from it.
template<class Real>
class OptimizableInterface {
 public:
  /// computes gradient for a parameter params and returns it
  /// in gradient_out
  virtual void ComputeGradient(const Vector<Real> &params,
                               Vector<Real> *gradient_out) = 0;
  /// computes the function value for a parameter params
  /// and returns it
  virtual Real ComputeValue(const Vector<Real> &params) = 0;

  virtual ~OptimizableInterface() {}
};
/// @} end of "Interfaces"
} // end namespace kaldi

#endif
