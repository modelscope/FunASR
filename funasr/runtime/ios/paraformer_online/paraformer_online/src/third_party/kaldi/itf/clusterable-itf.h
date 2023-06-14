// itf/clusterable-itf.h

// Copyright 2009-2011     Microsoft Corporation;  Go Vivace Inc.

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


#ifndef KALDI_ITF_CLUSTERABLE_ITF_H_
#define KALDI_ITF_CLUSTERABLE_ITF_H_ 1

#include <string>
#include "base/kaldi-common.h"

namespace kaldi {


/** \addtogroup clustering_group
 @{
  A virtual class for clusterable objects; see \ref clustering for an
  explanation if its function.
*/



class Clusterable {
 public:
  /// \name Functions that must be overridden
  /// @{

  /// Return a copy of this object.
  virtual Clusterable *Copy() const = 0;
  /// Return the objective function associated with the stats
  /// [assuming ML estimation]
  virtual BaseFloat Objf() const = 0;
  /// Return the normalizer (typically, count) associated with the stats
  virtual BaseFloat Normalizer() const = 0;
  /// Set stats to empty.
  virtual void SetZero() = 0;
  /// Add other stats.
  virtual void Add(const Clusterable &other) = 0;
  /// Subtract other stats.
  virtual void Sub(const Clusterable &other) = 0;
  /// Scale the stats by a positive number f [not mandatory to supply this].
  virtual void Scale(BaseFloat f) {
    KALDI_ERR << "This Clusterable object does not implement Scale().";
  }

  /// Return a string that describes the inherited type. 
  virtual std::string Type() const = 0;

  /// Write data to stream.
  virtual void Write(std::ostream &os, bool binary) const = 0;

  /// Read data from a stream and return the corresponding object (const
  /// function; it's a class member because we need access to the vtable
  /// so generic code can read derived types).
  virtual Clusterable* ReadNew(std::istream &os, bool binary) const = 0;

  virtual ~Clusterable() {}

  /// @}

  /// \name Functions that have default implementations
  /// @{

  // These functions have default implementations (but may be overridden for
  // speed). Implementatons in tree/clusterable-classes.cc

  /// Return the objective function of the combined object this + other.
  virtual BaseFloat ObjfPlus(const Clusterable &other) const;
  /// Return the objective function of the subtracted object this - other.
  virtual BaseFloat ObjfMinus(const Clusterable &other) const;
  /// Return the objective function decrease from merging the two
  /// clusters, negated to be a positive number (or zero).
  virtual BaseFloat Distance(const Clusterable &other) const;
  /// @}

};
/// @} end of "ingroup clustering_group"

}  // end namespace kaldi

#endif  // KALDI_ITF_CLUSTERABLE_ITF_H_

