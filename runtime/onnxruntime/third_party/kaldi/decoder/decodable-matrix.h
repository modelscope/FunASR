// decoder/decodable-matrix.h

// Copyright 2009-2011  Microsoft Corporation
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

#ifndef KALDI_DECODER_DECODABLE_MATRIX_H_
#define KALDI_DECODER_DECODABLE_MATRIX_H_

#include <vector>

#include "base/kaldi-common.h"
#include "itf/decodable-itf.h"
#include "itf/transition-information.h"
#include "matrix/kaldi-matrix.h"

namespace kaldi {


class DecodableMatrixScaledMapped: public DecodableInterface {
 public:
  // This constructor creates an object that will not delete "likes" when done.
  DecodableMatrixScaledMapped(const TransitionInformation &tm,
                              const Matrix<BaseFloat> &likes,
                              BaseFloat scale): trans_model_(tm), likes_(&likes),
                                                tid_to_pdf_(trans_model_.TransitionIdToPdfArray()),
                                                scale_(scale), delete_likes_(false) {
    if (likes.NumCols() != tm.NumPdfs())
      KALDI_ERR << "DecodableMatrixScaledMapped: mismatch, matrix has "
                << likes.NumCols() << " cols but transition-model has "
                << tm.NumPdfs() << " pdf-ids.";
  }

  // This constructor creates an object that will delete "likes"
  // when done.
  DecodableMatrixScaledMapped(const TransitionInformation &tm,
                              BaseFloat scale,
                              const Matrix<BaseFloat> *likes):
      trans_model_(tm), likes_(likes),
      tid_to_pdf_(trans_model_.TransitionIdToPdfArray()),
      scale_(scale), delete_likes_(true) {
    if (likes->NumCols() != tm.NumPdfs())
      KALDI_ERR << "DecodableMatrixScaledMapped: mismatch, matrix has "
                << likes->NumCols() << " cols but transition-model has "
                << tm.NumPdfs() << " pdf-ids.";
  }

  virtual int32 NumFramesReady() const { return likes_->NumRows(); }

  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1);
  }

  // Note, frames are numbered from zero.
  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    KALDI_PARANOID_ASSERT(tid >= 1 && tid < tid_to_pdf_.size());
    return scale_ * (*likes_)(frame, tid_to_pdf_[tid]);
  }

  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() const { return trans_model_.NumTransitionIds(); }

  virtual ~DecodableMatrixScaledMapped() {
    if (delete_likes_) delete likes_;
  }
 private:
  const TransitionInformation &trans_model_;  // for tid to pdf mapping
  const Matrix<BaseFloat> *likes_;
  const std::vector<int32> &tid_to_pdf_;
  BaseFloat scale_;
  bool delete_likes_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableMatrixScaledMapped);
};

/**
   This is like DecodableMatrixScaledMapped, but it doesn't support an acoustic
   scale, and it does support a frame offset, whereby you can state that the
   first row of 'likes' is actually the n'th row of the matrix of available
   log-likelihoods.  It's useful if the neural net output comes in chunks for
   different frame ranges.

   Note: DecodableMatrixMappedOffset solves the same problem in a slightly
   different way, where you use the same decodable object.  This one, unlike
   DecodableMatrixMappedOffset, is compatible with when the loglikes are in a
   SubMatrix.
 */
class DecodableMatrixMapped: public DecodableInterface {
 public:
  // This constructor creates an object that will not delete "likes" when done.
  // the frame_offset is the frame the row 0 of 'likes' corresponds to, would be
  // greater than one if this is not the first chunk of likelihoods.
  DecodableMatrixMapped(const TransitionInformation &tm,
                        const MatrixBase<BaseFloat> &likes,
                        int32 frame_offset = 0);

  // This constructor creates an object that will delete "likes"
  // when done.
  DecodableMatrixMapped(const TransitionInformation &tm,
                        const Matrix<BaseFloat> *likes,
                        int32 frame_offset = 0);

  virtual int32 NumFramesReady() const;

  virtual bool IsLastFrame(int32 frame) const;

  virtual BaseFloat LogLikelihood(int32 frame, int32 tid);

  // Note: these indices are 1-based.
  virtual int32 NumIndices() const;

  virtual ~DecodableMatrixMapped();

 private:
  const TransitionInformation &trans_model_;  // for tid to pdf mapping
  const std::vector<int32>& tid_to_pdf_;
  const MatrixBase<BaseFloat> *likes_;
  const Matrix<BaseFloat> *likes_to_delete_;
  int32 frame_offset_;

  // raw_data_ and stride_ are a kind of fast look-aside for 'likes_', to be
  // used when KALDI_PARANOID is false.
  const BaseFloat *raw_data_;
  int32 stride_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableMatrixMapped);
};


/**
   This decodable class returns log-likes stored in a matrix; it supports
   repeatedly writing to the matrix and setting a time-offset representing the
   frame-index of the first row of the matrix.  It's intended for use in
   multi-threaded decoding; mutex and semaphores are not included.  External
   code will call SetLoglikes() each time more log-likelihods are available.
   If you try to access a log-likelihood that's no longer available because
   the frame index is less than the current offset, it is of course an error.

   See also DecodableMatrixMapped, which supports the same type of thing but
   with a different interface where you are expected to re-construct the
   object each time you want to decode.
*/
class DecodableMatrixMappedOffset: public DecodableInterface {
 public:
  DecodableMatrixMappedOffset(const TransitionInformation &tm):
      trans_model_(tm), tid_to_pdf_(trans_model_.TransitionIdToPdfArray()),
      frame_offset_(0), input_is_finished_(false) { }

  // this is not part of the generic Decodable interface.
  int32 FirstAvailableFrame() const { return frame_offset_; }

  // Logically, this function appends 'loglikes' (interpreted as newly available
  // frames) to the log-likelihoods stored in the class.
  //
  // This function is destructive of the input "loglikes" because it may
  // under some circumstances do a shallow copy using Swap().  This function
  // appends loglikes to any existing likelihoods you've previously supplied.
  void AcceptLoglikes(Matrix<BaseFloat> *loglikes,
                      int32 frames_to_discard);

  void InputIsFinished() { input_is_finished_ = true; }

  virtual int32 NumFramesReady() const {
    return loglikes_.NumRows() + frame_offset_;
  }

  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1 && input_is_finished_);
  }

  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    KALDI_PARANOID_ASSERT(tid >= 1 && tid < tid_to_pdf_.size());
    int32 pdf_id = tid_to_pdf_[tid];
#ifdef KALDI_PARANOID
    return loglikes_(frame - frame_offset_, pdf_id);
#else
    // This does no checking, so will be faster.
    return raw_data_[frame * stride_ + pdf_id];
#endif
  }

  virtual int32 NumIndices() const { return trans_model_.NumTransitionIds(); }

  // nothing special to do in destructor.
  virtual ~DecodableMatrixMappedOffset() { }
 private:
  const TransitionInformation &trans_model_;  // for tid to pdf mapping
  const std::vector<int32>& tid_to_pdf_;
  Matrix<BaseFloat> loglikes_;
  int32 frame_offset_;
  bool input_is_finished_;

  // 'raw_data_' and 'stride_' are intended as a fast look-aside which is an
  // alternative to accessing data_.  raw_data_ is a faked version of
  // data_->Data() as if it started from frame zero rather than frame_offset_.
  // This simplifies the code of LogLikelihood(), in cases where KALDI_PARANOID
  // is not defined.
  BaseFloat *raw_data_;
  int32 stride_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableMatrixMappedOffset);
};


class DecodableMatrixScaled: public DecodableInterface {
 public:
  DecodableMatrixScaled(const Matrix<BaseFloat> &likes,
                        BaseFloat scale):
    likes_(likes), scale_(scale) { }

  virtual int32 NumFramesReady() const { return likes_.NumRows(); }

  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1);
  }

  // Note, frames are numbered from zero.
  virtual BaseFloat LogLikelihood(int32 frame, int32 index) {
    if (index > likes_.NumCols() || index <= 0 ||
        frame < 0 || frame >= likes_.NumRows())
      KALDI_ERR << "Invalid (frame, index - 1) = ("
                << frame << ", " << index - 1 << ") for matrix of size "
                << likes_.NumRows() << " x " << likes_.NumCols();
    return scale_ * likes_(frame, index - 1);
  }

  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() const { return likes_.NumCols(); }

 private:
  const Matrix<BaseFloat> &likes_;
  BaseFloat scale_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableMatrixScaled);
};
}  // namespace kaldi

#endif  // KALDI_DECODER_DECODABLE_MATRIX_H_
