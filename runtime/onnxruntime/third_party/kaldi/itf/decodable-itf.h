// itf/decodable-itf.h

// Copyright 2009-2011  Microsoft Corporation;  Saarland University;
//                      Mirko Hannemann;  Go Vivace Inc.;
//                2013  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_ITF_DECODABLE_ITF_H_
#define KALDI_ITF_DECODABLE_ITF_H_ 1
#include "base/kaldi-common.h"

namespace kaldi {
/// @ingroup Interfaces
/// @{


/**
    DecodableInterface provides a link between the (acoustic-modeling and
    feature-processing) code and the decoder.  The idea is to make this
    interface as small as possible, and to make it as agnostic as possible about
    the form of the acoustic model (e.g. don't assume the probabilities are a
    function of just a vector of floats), and about the decoder (e.g. don't
    assume it accesses frames in strict left-to-right order).  For normal
    models, without on-line operation, the "decodable" sub-class will just be a
    wrapper around a matrix of features and an acoustic model, and it will
    answer the question 'what is the acoustic likelihood for this index and this
    frame?'.

    For online decoding, where the features are coming in in real time, it is
    important to understand the IsLastFrame() and NumFramesReady() functions.
    There are two ways these are used: the old online-decoding code, in ../online/,
    and the new online-decoding code, in ../online2/.  In the old online-decoding
    code, the decoder would do:
    \code{.cc}
    for (int frame = 0; !decodable.IsLastFrame(frame); frame++) {
      // Process this frame
    }
    \endcode
   and the call to IsLastFrame would block if the features had not arrived yet.
   The decodable object would have to know when to terminate the decoding.  This
   online-decoding mode is still supported, it is what happens when you call, for
   example, LatticeFasterDecoder::Decode().

   We realized that this "blocking" mode of decoding is not very convenient
   because it forces the program to be multi-threaded and makes it complex to
   control endpointing.  In the "new" decoding code, you don't call (for example)
   LatticeFasterDecoder::Decode(), you call LatticeFasterDecoder::InitDecoding(),
   and then each time you get more features, you provide them to the decodable
   object, and you call LatticeFasterDecoder::AdvanceDecoding(), which does
   something like this:
   \code{.cc}
   while (num_frames_decoded_ < decodable.NumFramesReady()) {
     // Decode one more frame [increments num_frames_decoded_]
   }
   \endcode
   So the decodable object never has IsLastFrame() called.  For decoding where
   you are starting with a matrix of features, the NumFramesReady() function will
   always just return the number of frames in the file, and IsLastFrame() will
   return true for the last frame.

   For truly online decoding, the "old" online decodable objects in ../online/
   have a "blocking" IsLastFrame() and will crash if you call NumFramesReady().
   The "new" online decodable objects in ../online2/ return the number of frames
   currently accessible if you call NumFramesReady().  You will likely not need
   to call IsLastFrame(), but we implement it to only return true for the last
   frame of the file once we've decided to terminate decoding.
*/
class DecodableInterface {
 public:
  /// Returns the log likelihood, which will be negated in the decoder.
  /// The "frame" starts from zero.  You should verify that NumFramesReady() > frame
  /// before calling this.
  virtual BaseFloat LogLikelihood(int32 frame, int32 index) = 0;

  /// Returns true if this is the last frame.  Frames are zero-based, so the
  /// first frame is zero.  IsLastFrame(-1) will return false, unless the file
  /// is empty (which is a case that I'm not sure all the code will handle, so
  /// be careful).  Caution: the behavior of this function in an online setting
  /// is being changed somewhat.  In future it may return false in cases where
  /// we haven't yet decided to terminate decoding, but later true if we decide
  /// to terminate decoding.  The plan in future is to rely more on
  /// NumFramesReady(), and in future, IsLastFrame() would always return false
  /// in an online-decoding setting, and would only return true in a
  /// decoding-from-matrix setting where we want to allow the last delta or LDA
  /// features to be flushed out for compatibility with the baseline setup.
  virtual bool IsLastFrame(int32 frame) const = 0;

  /// The call NumFramesReady() will return the number of frames currently available
  /// for this decodable object.  This is for use in setups where you don't want the
  /// decoder to block while waiting for input.  This is newly added as of Jan 2014,
  /// and I hope, going forward, to rely on this mechanism more than IsLastFrame to
  /// know when to stop decoding.
  virtual int32 NumFramesReady() const {
    KALDI_ERR << "NumFramesReady() not implemented for this decodable type.";
    return -1;
  }

  /// Returns the number of states in the acoustic model
  /// (they will be indexed one-based, i.e. from 1 to NumIndices();
  /// this is for compatibility with OpenFst).
  virtual int32 NumIndices() const = 0;

  virtual ~DecodableInterface() {}
};
/// @}
}  // namespace Kaldi

#endif  // KALDI_ITF_DECODABLE_ITF_H_
