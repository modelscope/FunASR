// itf/online-feature-itf.h

// Copyright    2013  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_ITF_ONLINE_FEATURE_ITF_H_
#define KALDI_ITF_ONLINE_FEATURE_ITF_H_ 1
#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"

namespace kaldi {
/// @ingroup Interfaces
/// @{

/**
   OnlineFeatureInterface is an interface for online feature processing (it is
   also usable in the offline setting, but currently we're not using it for
   that).  This is for use in the online2/ directory, and it supersedes the
   interface in ../online/online-feat-input.h.  We have a slighty different
   model that puts more control in the hands of the calling thread, and won't
   involve waiting on semaphores in the decoding thread.

   This interface only specifies how the object *outputs* the features.
   How it obtains the features, e.g. from a previous object or objects of type
   OnlineFeatureInterface, is not specified in the interface and you will
   likely define new constructors or methods in the derived type to do that.

   You should appreciate that this interface is designed to allow random
   access to features, as long as they are ready.  That is, the user
   can call GetFrame for any frame less than NumFramesReady(), and when
   implementing a child class you must not make assumptions about the
   order in which the user makes these calls.
*/
   
class OnlineFeatureInterface {
 public:
  virtual int32 Dim() const = 0; /// returns the feature dimension.
  
  /// Returns the total number of frames, since the start of the utterance, that
  /// are now available.  In an online-decoding context, this will likely
  /// increase with time as more data becomes available.
  virtual int32 NumFramesReady() const = 0;

  /// Returns true if this is the last frame.  Frame indices are zero-based, so the
  /// first frame is zero.  IsLastFrame(-1) will return false, unless the file
  /// is empty (which is a case that I'm not sure all the code will handle, so
  /// be careful).  This function may return false for some frame if
  /// we haven't yet decided to terminate decoding, but later true if we decide
  /// to terminate decoding.  This function exists mainly to correctly handle
  /// end effects in feature extraction, and is not a mechanism to determine how
  /// many frames are in the decodable object (as it used to be, and for backward
  /// compatibility, still is, in the Decodable interface).
  virtual bool IsLastFrame(int32 frame) const = 0;
  
  /// Gets the feature vector for this frame.  Before calling this for a given
  /// frame, it is assumed that you called NumFramesReady() and it returned a
  /// number greater than "frame".  Otherwise this call will likely crash with
  /// an assert failure.  This function is not declared const, in case there is
  /// some kind of caching going on, but most of the time it shouldn't modify
  /// the class.
  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat) = 0;

  // Returns frame shift in seconds.  Helps to estimate duration from frame
  // counts.
  virtual BaseFloat FrameShiftInSeconds() const = 0;

  /// Virtual destructor.  Note: constructors that take another member of
  /// type OnlineFeatureInterface are not expected to take ownership of
  /// that pointer; the caller needs to keep track of that manually.
  virtual ~OnlineFeatureInterface() { }  
  
};


/// Add a virtual class for "source" features such as MFCC or PLP or pitch
/// features.
class OnlineBaseFeature: public OnlineFeatureInterface {
 public:
  /// This would be called from the application, when you get more wave data.
  /// Note: the sampling_rate is typically only provided so the code can assert
  /// that it matches the sampling rate expected in the options.
  virtual void AcceptWaveform(BaseFloat sampling_rate,
                              const VectorBase<BaseFloat> &waveform) = 0;

  /// InputFinished() tells the class you won't be providing any
  /// more waveform.  This will help flush out the last few frames
  /// of delta or LDA features (it will typically affect the return value
  /// of IsLastFrame.
  virtual void InputFinished() = 0;
};


/// @}
}  // namespace Kaldi

#endif  // KALDI_ITF_ONLINE_FEATURE_ITF_H_
