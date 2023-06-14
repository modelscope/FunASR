// matrix/sp-matrix-inl.h

// Copyright 2009-2011  Ondrej Glembek;  Microsoft Corporation;  Haihua Xu

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

#ifndef KALDI_MATRIX_SP_MATRIX_INL_H_
#define KALDI_MATRIX_SP_MATRIX_INL_H_

#include "matrix/tp-matrix.h"

namespace kaldi {

// All the lines in this file seem to be declaring template specializations.
// These tell the compiler that we'll implement the templated function
// separately for the different template arguments (float, double).

template<>
double SolveQuadraticProblem(const SpMatrix<double> &H, const VectorBase<double> &g,
                             const SolverOptions &opts, VectorBase<double> *x);

template<>
float SolveQuadraticProblem(const SpMatrix<float> &H, const VectorBase<float> &g,
                            const SolverOptions &opts, VectorBase<float> *x);

}  // namespace kaldi


#endif  // KALDI_MATRIX_SP_MATRIX_INL_H_
