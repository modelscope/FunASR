#ifndef WFST_DECODER_
#define WFST_DECODER_
#include "kaldi/decoder/lattice-faster-online-decoder.h"
#include "model.h"
#include "fst/fstlib.h"
#include "fst/symbol-table.h"
#include "bias-lm.h"
#include "phone-set.h"
#include "util.h"

#define MAX_SCORE 10.0f
namespace funasr {
class Decodable : public kaldi::DecodableInterface {
 public:
  Decodable(float scale = 1.0f) : scale_(scale) { 
    Reset(); 
  }
  void Reset() {
    num_frames_ = 0;
    finished_ = false;
    logp_.clear();
  }

  int NumFramesReady() const { return num_frames_; }

  bool IsLastFrame(int frame) const {
    return finished_ && (frame == num_frames_ - 1);
  }

  float LogLikelihood(int frm, int id) {
    CHECK_GT(id, 0);
    CHECK_LT(frm, num_frames_);
    return scale_ * logp_[id - 1];
  }

  void AcceptLoglikes(const std::vector<float>& logp) {
    num_frames_++;
    logp_ = logp;
  }

  int NumIndices() const { return 0; }
  void SetFinished() { finished_ = true; }

 private:
  int num_frames_ = 0;
  float scale_ = 1.0f;
  bool finished_ = false;
  std::vector<float> logp_;
};

struct DecodeOptions : public kaldi::LatticeFasterDecoderConfig {
  DecodeOptions(float glob_beam = 3.0f, float lat_beam = 3.0f, float ac_sc = 10.0f) :
  kaldi::LatticeFasterDecoderConfig(glob_beam, lat_beam), acoustic_scale(ac_sc) {
  }
  float acoustic_scale;
};

class WfstDecoder {
 public:
  WfstDecoder(fst::Fst<fst::StdArc>* lm,
              PhoneSet* phone_set,
              Vocab* vocab,
              float glob_beam,
              float lat_beam,
              float am_scale);
  ~WfstDecoder();
  void StartUtterance();
  void EndUtterance();
  string Search(float *in, int len, int64_t token_nums);
  string FinalizeDecode(bool is_stamp=false, std::vector<float> us_alphas={0}, std::vector<float> us_cif_peak={0});
  void LoadHwsRes(int inc_bias, unordered_map<string, int> &hws_map);
  void UnloadHwsRes();

 private:
  Vocab* vocab_ = nullptr;
  PhoneSet* phone_set_ = nullptr;
  int cur_frame_ = 0;
  int cur_token_ = 0;
  DecodeOptions dec_opts_;
  Decodable decodable_;
  fst::Fst<fst::StdArc>* lm_ = nullptr;
  std::shared_ptr<kaldi::LatticeFasterOnlineDecoder> decoder_ = nullptr;
  std::shared_ptr<BiasLm> bias_lm_ = nullptr;
};
} // namespace funasr
#endif // WFST_DECODER_
