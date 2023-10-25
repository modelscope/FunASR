// decoder/decodable-matrix.cc

// Copyright    2018 Johns Hopkins University (author: Daniel Povey)

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

#include "decoder/decodable-matrix.h"

namespace kaldi {

DecodableMatrixMapped::DecodableMatrixMapped(
    const TransitionInformation &tm,
    const MatrixBase<BaseFloat> &likes,
    int32 frame_offset):
    trans_model_(tm),
    tid_to_pdf_(trans_model_.TransitionIdToPdfArray()),
    likes_(&likes), likes_to_delete_(NULL),
    frame_offset_(frame_offset) {
  stride_ = likes.Stride();
  raw_data_ = likes.Data() - (stride_ * frame_offset);

  if (likes.NumCols() != tm.NumPdfs())
    KALDI_ERR << "Mismatch, matrix has "
              << likes.NumCols() << " cols but transition-model has "
              << tm.NumPdfs() << " pdf-ids.";
}

DecodableMatrixMapped::DecodableMatrixMapped(
    const TransitionInformation &tm, const Matrix<BaseFloat> *likes,
    int32 frame_offset):
    trans_model_(tm),
    tid_to_pdf_(trans_model_.TransitionIdToPdfArray()),
    likes_(likes), likes_to_delete_(likes),
    frame_offset_(frame_offset) {
  stride_ = likes->Stride();
  raw_data_ = likes->Data() - (stride_ * frame_offset_);
  if (likes->NumCols() != tm.NumPdfs())
    KALDI_ERR << "Mismatch, matrix has "
              << likes->NumCols() << " cols but transition-model has "
              << tm.NumPdfs() << " pdf-ids.";
}


BaseFloat DecodableMatrixMapped::LogLikelihood(int32 frame, int32 tid) {
  KALDI_PARANOID_ASSERT(tid >= 1 && tid < tid_to_pdf_.size());
  int32 pdf_id = tid_to_pdf_[tid];
#ifdef KALDI_PARANOID
  return (*likes_)(frame - frame_offset_, pdf_id);
#else
  return raw_data_[frame * stride_ + pdf_id];
#endif
}

int32 DecodableMatrixMapped::NumFramesReady() const {
  return frame_offset_ + likes_->NumRows();
}

bool DecodableMatrixMapped::IsLastFrame(int32 frame) const {
  KALDI_ASSERT(frame < NumFramesReady());
  return (frame == NumFramesReady() - 1);
}

// Indices are one-based!  This is for compatibility with OpenFst.
int32 DecodableMatrixMapped::NumIndices() const {
  return trans_model_.NumTransitionIds();
}

DecodableMatrixMapped::~DecodableMatrixMapped() {
  delete likes_to_delete_;
}


void DecodableMatrixMappedOffset::AcceptLoglikes(
    Matrix<BaseFloat> *loglikes, int32 frames_to_discard) {
  if (loglikes->NumRows() == 0) return;
  KALDI_ASSERT(loglikes->NumCols() == trans_model_.NumPdfs());
  KALDI_ASSERT(frames_to_discard <= loglikes_.NumRows() &&
               frames_to_discard >= 0);
  if (frames_to_discard == loglikes_.NumRows()) {
    loglikes_.Swap(loglikes);
    loglikes->Resize(0, 0);
  } else {
    int32 old_rows_kept = loglikes_.NumRows() - frames_to_discard,
        new_num_rows = old_rows_kept + loglikes->NumRows();
    Matrix<BaseFloat> new_loglikes(new_num_rows, loglikes->NumCols());
    new_loglikes.RowRange(0, old_rows_kept).CopyFromMat(
        loglikes_.RowRange(frames_to_discard, old_rows_kept));
    new_loglikes.RowRange(old_rows_kept, loglikes->NumRows()).CopyFromMat(
        *loglikes);
    loglikes_.Swap(&new_loglikes);
  }
  frame_offset_ += frames_to_discard;
  stride_ = loglikes_.Stride();
  raw_data_ = loglikes_.Data() - (frame_offset_ * stride_);
}



} // end namespace kaldi.
