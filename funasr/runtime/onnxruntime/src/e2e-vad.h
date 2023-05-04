/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
 * Collaborators: zhuzizyf(China Telecom Shanghai)
*/

#include <utility>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cassert>


enum class VadStateMachine {
    kVadInStateStartPointNotDetected = 1,
    kVadInStateInSpeechSegment = 2,
    kVadInStateEndPointDetected = 3
};

enum class FrameState {
    kFrameStateInvalid = -1,
    kFrameStateSpeech = 1,
    kFrameStateSil = 0
};

// final voice/unvoice state per frame
enum class AudioChangeState {
    kChangeStateSpeech2Speech = 0,
    kChangeStateSpeech2Sil = 1,
    kChangeStateSil2Sil = 2,
    kChangeStateSil2Speech = 3,
    kChangeStateNoBegin = 4,
    kChangeStateInvalid = 5
};

enum class VadDetectMode {
    kVadSingleUtteranceDetectMode = 0,
    kVadMutipleUtteranceDetectMode = 1
};

class VADXOptions {
public:
    int sample_rate;
    int detect_mode;
    int snr_mode;
    int max_end_silence_time;
    int max_start_silence_time;
    bool do_start_point_detection;
    bool do_end_point_detection;
    int window_size_ms;
    int sil_to_speech_time_thres;
    int speech_to_sil_time_thres;
    float speech_2_noise_ratio;
    int do_extend;
    int lookback_time_start_point;
    int lookahead_time_end_point;
    int max_single_segment_time;
    int nn_eval_block_size;
    int dcd_block_size;
    float snr_thres;
    int noise_frame_num_used_for_snr;
    float decibel_thres;
    float speech_noise_thres;
    float fe_prior_thres;
    int silence_pdf_num;
    std::vector<int> sil_pdf_ids;
    float speech_noise_thresh_low;
    float speech_noise_thresh_high;
    bool output_frame_probs;
    int frame_in_ms;
    int frame_length_ms;

    explicit VADXOptions(
            int sr = 16000,
            int dm = static_cast<int>(VadDetectMode::kVadMutipleUtteranceDetectMode),
            int sm = 0,
            int mset = 800,
            int msst = 3000,
            bool dspd = true,
            bool depd = true,
            int wsm = 200,
            int ststh = 150,
            int sttsh = 150,
            float s2nr = 1.0,
            int de = 1,
            int lbtps = 200,
            int latsp = 100,
            int mss = 15000,
            int nebs = 8,
            int dbs = 4,
            float st = -100.0,
            int nfnus = 100,
            float dt = -100.0,
            float snt = 0.9,
            float fept = 1e-4,
            int spn = 1,
            std::vector<int> spids = {0},
            float sntl = -0.1,
            float snth = 0.3,
            bool ofp = false,
            int fim = 10,
            int flm = 25
    ) :
            sample_rate(sr),
            detect_mode(dm),
            snr_mode(sm),
            max_end_silence_time(mset),
            max_start_silence_time(msst),
            do_start_point_detection(dspd),
            do_end_point_detection(depd),
            window_size_ms(wsm),
            sil_to_speech_time_thres(ststh),
            speech_to_sil_time_thres(sttsh),
            speech_2_noise_ratio(s2nr),
            do_extend(de),
            lookback_time_start_point(lbtps),
            lookahead_time_end_point(latsp),
            max_single_segment_time(mss),
            nn_eval_block_size(nebs),
            dcd_block_size(dbs),
            snr_thres(st),
            noise_frame_num_used_for_snr(nfnus),
            decibel_thres(dt),
            speech_noise_thres(snt),
            fe_prior_thres(fept),
            silence_pdf_num(spn),
            sil_pdf_ids(std::move(spids)),
            speech_noise_thresh_low(sntl),
            speech_noise_thresh_high(snth),
            output_frame_probs(ofp),
            frame_in_ms(fim),
            frame_length_ms(flm) {}
};

class E2EVadSpeechBufWithDoa {
public:
    int start_ms;
    int end_ms;
    std::vector<float> buffer;
    bool contain_seg_start_point;
    bool contain_seg_end_point;
    int doa;

    E2EVadSpeechBufWithDoa() :
            start_ms(0),
            end_ms(0),
            buffer(),
            contain_seg_start_point(false),
            contain_seg_end_point(false),
            doa(0) {}

    void Reset() {
        start_ms = 0;
        end_ms = 0;
        buffer.clear();
        contain_seg_start_point = false;
        contain_seg_end_point = false;
        doa = 0;
    }
};

class E2EVadFrameProb {
public:
    double noise_prob;
    double speech_prob;
    double score;
    int frame_id;
    int frm_state;

    E2EVadFrameProb() :
            noise_prob(0.0),
            speech_prob(0.0),
            score(0.0),
            frame_id(0),
            frm_state(0) {}
};

class WindowDetector {
public:
    int window_size_ms;
    int sil_to_speech_time;
    int speech_to_sil_time;
    int frame_size_ms;
    int win_size_frame;
    int win_sum;
    std::vector<int> win_state;
    int cur_win_pos;
    FrameState pre_frame_state;
    FrameState cur_frame_state;
    int sil_to_speech_frmcnt_thres;
    int speech_to_sil_frmcnt_thres;
    int voice_last_frame_count;
    int noise_last_frame_count;
    int hydre_frame_count;

    WindowDetector(int window_size_ms, int sil_to_speech_time, int speech_to_sil_time, int frame_size_ms) :
            window_size_ms(window_size_ms),
            sil_to_speech_time(sil_to_speech_time),
            speech_to_sil_time(speech_to_sil_time),
            frame_size_ms(frame_size_ms),
            win_size_frame(window_size_ms / frame_size_ms),
            win_sum(0),
            win_state(std::vector<int>(win_size_frame, 0)),
            cur_win_pos(0),
            pre_frame_state(FrameState::kFrameStateSil),
            cur_frame_state(FrameState::kFrameStateSil),
            sil_to_speech_frmcnt_thres(sil_to_speech_time / frame_size_ms),
            speech_to_sil_frmcnt_thres(speech_to_sil_time / frame_size_ms),
            voice_last_frame_count(0),
            noise_last_frame_count(0),
            hydre_frame_count(0) {}

    void Reset() {
        cur_win_pos = 0;
        win_sum = 0;
        win_state = std::vector<int>(win_size_frame, 0);
        pre_frame_state = FrameState::kFrameStateSil;
        cur_frame_state = FrameState::kFrameStateSil;
        voice_last_frame_count = 0;
        noise_last_frame_count = 0;
        hydre_frame_count = 0;
    }

    int GetWinSize() {
        return win_size_frame;
    }

    AudioChangeState DetectOneFrame(FrameState frameState, int frame_count) {
        int cur_frame_state = 0;
        if (frameState == FrameState::kFrameStateSpeech) {
            cur_frame_state = 1;
        } else if (frameState == FrameState::kFrameStateSil) {
            cur_frame_state = 0;
        } else {
            return AudioChangeState::kChangeStateInvalid;
        }
        win_sum -= win_state[cur_win_pos];
        win_sum += cur_frame_state;
        win_state[cur_win_pos] = cur_frame_state;
        cur_win_pos = (cur_win_pos + 1) % win_size_frame;
        if (pre_frame_state == FrameState::kFrameStateSil && win_sum >= sil_to_speech_frmcnt_thres) {
            pre_frame_state = FrameState::kFrameStateSpeech;
            return AudioChangeState::kChangeStateSil2Speech;
        }
        if (pre_frame_state == FrameState::kFrameStateSpeech && win_sum <= speech_to_sil_frmcnt_thres) {
            pre_frame_state = FrameState::kFrameStateSil;
            return AudioChangeState::kChangeStateSpeech2Sil;
        }
        if (pre_frame_state == FrameState::kFrameStateSil) {
            return AudioChangeState::kChangeStateSil2Sil;
        }
        if (pre_frame_state == FrameState::kFrameStateSpeech) {
            return AudioChangeState::kChangeStateSpeech2Speech;
        }
        return AudioChangeState::kChangeStateInvalid;
    }

    int FrameSizeMs() {
        return frame_size_ms;
    }
};

class E2EVadModel {
public:
    E2EVadModel() {
        this->vad_opts = VADXOptions();
//    this->windows_detector = WindowDetector(200,150,150,10);
        // this->encoder = encoder;
        // init variables
        this->is_final = false;
        this->data_buf_start_frame = 0;
        this->frm_cnt = 0;
        this->latest_confirmed_speech_frame = 0;
        this->lastest_confirmed_silence_frame = -1;
        this->continous_silence_frame_count = 0;
        this->vad_state_machine = VadStateMachine::kVadInStateStartPointNotDetected;
        this->confirmed_start_frame = -1;
        this->confirmed_end_frame = -1;
        this->number_end_time_detected = 0;
        this->sil_frame = 0;
        this->sil_pdf_ids = this->vad_opts.sil_pdf_ids;
        this->noise_average_decibel = -100.0;
        this->pre_end_silence_detected = false;
        this->next_seg = true;
//    this->output_data_buf = [];
        this->output_data_buf_offset = 0;
//    this->frame_probs = [];
        this->max_end_sil_frame_cnt_thresh =
                this->vad_opts.max_end_silence_time - this->vad_opts.speech_to_sil_time_thres;
        this->speech_noise_thres = this->vad_opts.speech_noise_thres;
        this->max_time_out = false;
//    this->decibel = [];
        this->ResetDetection();
    }

    std::vector<std::vector<int>>
    operator()(const std::vector<std::vector<float>> &score, const std::vector<float> &waveform, bool is_final = false,
               bool online = false, int max_end_sil = 800, int max_single_segment_time = 15000,
               float speech_noise_thres = 0.8, int sample_rate = 16000) {
        max_end_sil_frame_cnt_thresh = max_end_sil - vad_opts.speech_to_sil_time_thres;
        this->waveform = waveform;
        this->vad_opts.max_single_segment_time = max_single_segment_time;
        this->vad_opts.speech_noise_thres = speech_noise_thres;
        this->vad_opts.sample_rate = sample_rate;

        ComputeDecibel();
        ComputeScores(score);
        if (!is_final) {
            DetectCommonFrames();
        } else {
            DetectLastFrames();
        }

        std::vector<std::vector<int>> segment_batch;
        if (output_data_buf.size() > 0) {
            for (size_t i = output_data_buf_offset; i < output_data_buf.size(); i++) {
              int start_ms;
              int end_ms;
              if (online) {

                if (!output_data_buf[i].contain_seg_start_point) {
                  continue;
                }
                if (!next_seg && !output_data_buf[i].contain_seg_end_point) {
                  continue;
                }
                start_ms = next_seg ? output_data_buf[i].start_ms : -1;

                if (output_data_buf[i].contain_seg_end_point) {
                  end_ms = output_data_buf[i].end_ms;
                  next_seg = true;
                  output_data_buf_offset += 1;
                } else {
                  end_ms = -1;
                  next_seg = false;
                }
              } else {
                if (!is_final &&
                    (!output_data_buf[i].contain_seg_start_point || !output_data_buf[i].contain_seg_end_point)) {
                  continue;
                }
                start_ms = output_data_buf[i].start_ms;
                end_ms = output_data_buf[i].end_ms;
                output_data_buf_offset += 1;
              }
                std::vector<int> segment = {start_ms, end_ms};
                segment_batch.push_back(segment);
            }
        }

        if (is_final) {
            AllResetDetection();
        }
        return segment_batch;
    }

private:
    VADXOptions vad_opts;
    WindowDetector windows_detector = WindowDetector(200, 150, 150, 10);
    bool is_final;
    int data_buf_start_frame;
    int frm_cnt;
    int latest_confirmed_speech_frame;
    int lastest_confirmed_silence_frame;
    int continous_silence_frame_count;
    VadStateMachine vad_state_machine;
    int confirmed_start_frame;
    int confirmed_end_frame;
    int number_end_time_detected;
    int sil_frame;
    std::vector<int> sil_pdf_ids;
    float noise_average_decibel;
    bool pre_end_silence_detected;
    bool next_seg;
    std::vector<E2EVadSpeechBufWithDoa> output_data_buf;
    int output_data_buf_offset;
    std::vector<E2EVadFrameProb> frame_probs;
    int max_end_sil_frame_cnt_thresh;
    float speech_noise_thres;
    std::vector<std::vector<float>> scores;
    int idx_pre_chunk = 0;
    bool max_time_out;
    std::vector<float> decibel;
    int data_buf_size = 0;
    int data_buf_all_size = 0;
    std::vector<float> waveform;

    void AllResetDetection() {
        is_final = false;
        data_buf_start_frame = 0;
        frm_cnt = 0;
        latest_confirmed_speech_frame = 0;
        lastest_confirmed_silence_frame = -1;
        continous_silence_frame_count = 0;
        vad_state_machine = VadStateMachine::kVadInStateStartPointNotDetected;
        confirmed_start_frame = -1;
        confirmed_end_frame = -1;
        number_end_time_detected = 0;
        sil_frame = 0;
        sil_pdf_ids = vad_opts.sil_pdf_ids;
        noise_average_decibel = -100.0;
        pre_end_silence_detected = false;
        next_seg = true;
        output_data_buf.clear();
        output_data_buf_offset = 0;
        frame_probs.clear();
        max_end_sil_frame_cnt_thresh = vad_opts.max_end_silence_time - vad_opts.speech_to_sil_time_thres;
        speech_noise_thres = vad_opts.speech_noise_thres;
        scores.clear();
        idx_pre_chunk = 0;
        max_time_out = false;
        decibel.clear();
        int data_buf_size = 0;
        int data_buf_all_size = 0;
        waveform.clear();
        ResetDetection();
    }

    void ResetDetection() {
        continous_silence_frame_count = 0;
        latest_confirmed_speech_frame = 0;
        lastest_confirmed_silence_frame = -1;
        confirmed_start_frame = -1;
        confirmed_end_frame = -1;
        vad_state_machine = VadStateMachine::kVadInStateStartPointNotDetected;
        windows_detector.Reset();
        sil_frame = 0;
        frame_probs.clear();
    }

    void ComputeDecibel() {
        int frame_sample_length = int(vad_opts.frame_length_ms * vad_opts.sample_rate / 1000);
        int frame_shift_length = int(vad_opts.frame_in_ms * vad_opts.sample_rate / 1000);
        if (data_buf_all_size == 0) {
          data_buf_all_size = waveform.size();
          data_buf_size = data_buf_all_size;
        } else {
          data_buf_all_size += waveform.size();
        }
        for (int offset = 0; offset < waveform.size() - frame_sample_length + 1; offset += frame_shift_length) {
            float sum = 0.0;
            for (int i = 0; i < frame_sample_length; i++) {
                sum += waveform[offset + i] * waveform[offset + i];
            }
            this->decibel.push_back(10 * log10(sum + 0.000001));
        }
    }

    void ComputeScores(const std::vector<std::vector<float>> &scores) {
        vad_opts.nn_eval_block_size = scores.size();
        frm_cnt += scores.size();
        this->scores = scores;
    }

    void PopDataBufTillFrame(int frame_idx) {
      int frame_sample_length = int(vad_opts.frame_in_ms * vad_opts.sample_rate / 1000);
      while (data_buf_start_frame < frame_idx) {
        if (data_buf_size >= frame_sample_length) {
          data_buf_start_frame += 1;
          data_buf_size = data_buf_all_size - data_buf_start_frame * frame_sample_length;
        }
      }
    }

    void PopDataToOutputBuf(int start_frm, int frm_cnt, bool first_frm_is_start_point, bool last_frm_is_end_point,
                            bool end_point_is_sent_end) {
        PopDataBufTillFrame(start_frm);
        int expected_sample_number = int(frm_cnt * vad_opts.sample_rate * vad_opts.frame_in_ms / 1000);
        if (last_frm_is_end_point) {
            int extra_sample = std::max(0, int(vad_opts.frame_length_ms * vad_opts.sample_rate / 1000 -
                                               vad_opts.sample_rate * vad_opts.frame_in_ms / 1000));
            expected_sample_number += int(extra_sample);
        }
        if (end_point_is_sent_end) {
            expected_sample_number = std::max(expected_sample_number, data_buf_size);
        }
        if (data_buf_size < expected_sample_number) {
            std::cout << "error in calling pop data_buf\n";
        }
        if (output_data_buf.size() == 0 || first_frm_is_start_point) {
            output_data_buf.push_back(E2EVadSpeechBufWithDoa());
            output_data_buf[output_data_buf.size() - 1].Reset();
            output_data_buf[output_data_buf.size() - 1].start_ms = start_frm * vad_opts.frame_in_ms;
            output_data_buf[output_data_buf.size() - 1].end_ms = output_data_buf[output_data_buf.size() - 1].start_ms;
            output_data_buf[output_data_buf.size() - 1].doa = 0;
        }
        E2EVadSpeechBufWithDoa &cur_seg = output_data_buf.back();
        if (cur_seg.end_ms != start_frm * vad_opts.frame_in_ms) {
            std::cout << "warning\n";
        }
        int out_pos = (int) cur_seg.buffer.size();
        int data_to_pop;
        if (end_point_is_sent_end) {
            data_to_pop = expected_sample_number;
        } else {
            data_to_pop = int(frm_cnt * vad_opts.frame_in_ms * vad_opts.sample_rate / 1000);
        }
        if (data_to_pop > data_buf_size) {
            std::cout << "VAD data_to_pop is bigger than data_buf.size()!!!\n";
            data_to_pop = data_buf_size;
            expected_sample_number = data_buf_size;
        }
        cur_seg.doa = 0;
        for (int sample_cpy_out = 0; sample_cpy_out < data_to_pop; sample_cpy_out++) {
            cur_seg.buffer.push_back(data_buf.back());
            out_pos++;
        }
        for (int sample_cpy_out = data_to_pop; sample_cpy_out < expected_sample_number; sample_cpy_out++) {
            cur_seg.buffer.push_back(data_buf.back());
            out_pos++;
        }
        if (cur_seg.end_ms != start_frm * vad_opts.frame_in_ms) {
            std::cout << "Something wrong with the VAD algorithm\n";
        }
        data_buf_start_frame += frm_cnt;
        cur_seg.end_ms = (start_frm + frm_cnt) * vad_opts.frame_in_ms;
        if (first_frm_is_start_point) {
            cur_seg.contain_seg_start_point = true;
        }
        if (last_frm_is_end_point) {
            cur_seg.contain_seg_end_point = true;
        }
    }

    void OnSilenceDetected(int valid_frame) {
        lastest_confirmed_silence_frame = valid_frame;
        if (vad_state_machine == VadStateMachine::kVadInStateStartPointNotDetected) {
            PopDataBufTillFrame(valid_frame);
        }
        // silence_detected_callback_
        // pass
    }

    void OnVoiceDetected(int valid_frame) {
        latest_confirmed_speech_frame = valid_frame;
        PopDataToOutputBuf(valid_frame, 1, false, false, false);
    }

    void OnVoiceStart(int start_frame, bool fake_result = false) {
        if (vad_opts.do_start_point_detection) {
            // pass
        }
        if (confirmed_start_frame != -1) {
            std::cout << "not reset vad properly\n";
        } else {
            confirmed_start_frame = start_frame;
        }
        if (!fake_result && vad_state_machine == VadStateMachine::kVadInStateStartPointNotDetected) {
            PopDataToOutputBuf(confirmed_start_frame, 1, true, false, false);
        }
    }


    void OnVoiceEnd(int end_frame, bool fake_result, bool is_last_frame) {
        for (int t = latest_confirmed_speech_frame + 1; t < end_frame; t++) {
            OnVoiceDetected(t);
        }
        if (vad_opts.do_end_point_detection) {
            // pass
        }
        if (confirmed_end_frame != -1) {
            std::cout << "not reset vad properly\n";
        } else {
            confirmed_end_frame = end_frame;
        }
        if (!fake_result) {
            sil_frame = 0;
            PopDataToOutputBuf(confirmed_end_frame, 1, false, true, is_last_frame);
        }
        number_end_time_detected++;
    }

    void MaybeOnVoiceEndIfLastFrame(bool is_final_frame, int cur_frm_idx) {
        if (is_final_frame) {
            OnVoiceEnd(cur_frm_idx, false, true);
            vad_state_machine = VadStateMachine::kVadInStateEndPointDetected;
        }
    }

    int GetLatency() {
        return int(LatencyFrmNumAtStartPoint() * vad_opts.frame_in_ms);
    }

    int LatencyFrmNumAtStartPoint() {
        int vad_latency = windows_detector.GetWinSize();
        if (vad_opts.do_extend) {
            vad_latency += int(vad_opts.lookback_time_start_point / vad_opts.frame_in_ms);
        }
        return vad_latency;
    }

    FrameState GetFrameState(int t) {
        FrameState frame_state = FrameState::kFrameStateInvalid;
        float cur_decibel = decibel[t];
        float cur_snr = cur_decibel - noise_average_decibel;
        if (cur_decibel < vad_opts.decibel_thres) {
            frame_state = FrameState::kFrameStateSil;
            DetectOneFrame(frame_state, t, false);
            return frame_state;
        }
        float sum_score = 0.0;
        float noise_prob = 0.0;
        assert(sil_pdf_ids.size() == vad_opts.silence_pdf_num);
        if (sil_pdf_ids.size() > 0) {
            std::vector<float> sil_pdf_scores;
            for (auto sil_pdf_id: sil_pdf_ids) {
                sil_pdf_scores.push_back(scores[t - idx_pre_chunk][sil_pdf_id]);
            }
            sum_score = accumulate(sil_pdf_scores.begin(), sil_pdf_scores.end(), 0.0);
            noise_prob = log(sum_score) * vad_opts.speech_2_noise_ratio;
            float total_score = 1.0;
            sum_score = total_score - sum_score;
        }
        float speech_prob = log(sum_score);
        if (vad_opts.output_frame_probs) {
            E2EVadFrameProb frame_prob;
            frame_prob.noise_prob = noise_prob;
            frame_prob.speech_prob = speech_prob;
            frame_prob.score = sum_score;
            frame_prob.frame_id = t;
            frame_probs.push_back(frame_prob);
        }
        if (exp(speech_prob) >= exp(noise_prob) + speech_noise_thres) {
            if (cur_snr >= vad_opts.snr_thres && cur_decibel >= vad_opts.decibel_thres) {
                frame_state = FrameState::kFrameStateSpeech;
            } else {
                frame_state = FrameState::kFrameStateSil;
            }
        } else {
            frame_state = FrameState::kFrameStateSil;
            if (noise_average_decibel < -99.9) {
                noise_average_decibel = cur_decibel;
            } else {
                noise_average_decibel =
                        (cur_decibel + noise_average_decibel * (vad_opts.noise_frame_num_used_for_snr - 1)) /
                        vad_opts.noise_frame_num_used_for_snr;
            }
        }
        return frame_state;
    }

    int DetectCommonFrames() {
        if (vad_state_machine == VadStateMachine::kVadInStateEndPointDetected) {
            return 0;
        }
        for (int i = vad_opts.nn_eval_block_size - 1; i >= 0; i--) {
            FrameState frame_state = FrameState::kFrameStateInvalid;
            frame_state = GetFrameState(frm_cnt - 1 - i);
            DetectOneFrame(frame_state, frm_cnt - 1 - i, false);
        }
        idx_pre_chunk += scores.size();
        return 0;
    }

    int DetectLastFrames() {
        if (vad_state_machine == VadStateMachine::kVadInStateEndPointDetected) {
            return 0;
        }
        for (int i = vad_opts.nn_eval_block_size - 1; i >= 0; i--) {
            FrameState frame_state = FrameState::kFrameStateInvalid;
            frame_state = GetFrameState(frm_cnt - 1 - i);
            if (i != 0) {
                DetectOneFrame(frame_state, frm_cnt - 1 - i, false);
            } else {
                DetectOneFrame(frame_state, frm_cnt - 1, true);
            }
        }
        return 0;
    }

    void DetectOneFrame(FrameState cur_frm_state, int cur_frm_idx, bool is_final_frame) {
        FrameState tmp_cur_frm_state = FrameState::kFrameStateInvalid;
        if (cur_frm_state == FrameState::kFrameStateSpeech) {
            if (std::fabs(1.0) > vad_opts.fe_prior_thres) {
                tmp_cur_frm_state = FrameState::kFrameStateSpeech;
            } else {
                tmp_cur_frm_state = FrameState::kFrameStateSil;
            }
        } else if (cur_frm_state == FrameState::kFrameStateSil) {
            tmp_cur_frm_state = FrameState::kFrameStateSil;
        }
        AudioChangeState state_change = windows_detector.DetectOneFrame(tmp_cur_frm_state, cur_frm_idx);
        int frm_shift_in_ms = vad_opts.frame_in_ms;
        if (AudioChangeState::kChangeStateSil2Speech == state_change) {
            int silence_frame_count = continous_silence_frame_count;
            continous_silence_frame_count = 0;
            pre_end_silence_detected = false;
            int start_frame = 0;
            if (vad_state_machine == VadStateMachine::kVadInStateStartPointNotDetected) {
                start_frame = std::max(data_buf_start_frame, cur_frm_idx - LatencyFrmNumAtStartPoint());
                OnVoiceStart(start_frame);
                vad_state_machine = VadStateMachine::kVadInStateInSpeechSegment;
                for (int t = start_frame + 1; t <= cur_frm_idx; t++) {
                    OnVoiceDetected(t);
                }
            } else if (vad_state_machine == VadStateMachine::kVadInStateInSpeechSegment) {
                for (int t = latest_confirmed_speech_frame + 1; t < cur_frm_idx; t++) {
                    OnVoiceDetected(t);
                }
                if (cur_frm_idx - confirmed_start_frame + 1 > vad_opts.max_single_segment_time / frm_shift_in_ms) {
                    OnVoiceEnd(cur_frm_idx, false, false);
                    vad_state_machine = VadStateMachine::kVadInStateEndPointDetected;
                } else if (!is_final_frame) {
                    OnVoiceDetected(cur_frm_idx);
                } else {
                    MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx);
                }
            }
        } else if (AudioChangeState::kChangeStateSpeech2Sil == state_change) {
            continous_silence_frame_count = 0;
            if (vad_state_machine == VadStateMachine::kVadInStateStartPointNotDetected) {
                // do nothing
            } else if (vad_state_machine == VadStateMachine::kVadInStateInSpeechSegment) {
                if (cur_frm_idx - confirmed_start_frame + 1 >
                    vad_opts.max_single_segment_time / frm_shift_in_ms) {
                    OnVoiceEnd(cur_frm_idx, false, false);
                    vad_state_machine = VadStateMachine::kVadInStateEndPointDetected;
                } else if (!is_final_frame) {
                    OnVoiceDetected(cur_frm_idx);
                } else {
                    MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx);
                }
            }
        } else if (AudioChangeState::kChangeStateSpeech2Speech == state_change) {
            continous_silence_frame_count = 0;
            if (vad_state_machine == VadStateMachine::kVadInStateInSpeechSegment) {
                if (cur_frm_idx - confirmed_start_frame + 1 >
                    vad_opts.max_single_segment_time / frm_shift_in_ms) {
                    max_time_out = true;
                    OnVoiceEnd(cur_frm_idx, false, false);
                    vad_state_machine = VadStateMachine::kVadInStateEndPointDetected;
                } else if (!is_final_frame) {
                    OnVoiceDetected(cur_frm_idx);
                } else {
                    MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx);
                }
            }
        } else if (AudioChangeState::kChangeStateSil2Sil == state_change) {
            continous_silence_frame_count += 1;
            if (vad_state_machine == VadStateMachine::kVadInStateStartPointNotDetected) {
                if ((vad_opts.detect_mode == static_cast<int>(VadDetectMode::kVadSingleUtteranceDetectMode) &&
                     (continous_silence_frame_count * frm_shift_in_ms > vad_opts.max_start_silence_time)) ||
                    (is_final_frame && number_end_time_detected == 0)) {
                    for (int t = lastest_confirmed_silence_frame + 1; t < cur_frm_idx; t++) {
                        OnSilenceDetected(t);
                    }
                    OnVoiceStart(0, true);
                    OnVoiceEnd(0, true, false);
                    vad_state_machine = VadStateMachine::kVadInStateEndPointDetected;
                } else {
                    if (cur_frm_idx >= LatencyFrmNumAtStartPoint()) {
                        OnSilenceDetected(cur_frm_idx - LatencyFrmNumAtStartPoint());
                    }
                }
            } else if (vad_state_machine == VadStateMachine::kVadInStateInSpeechSegment) {
                if (continous_silence_frame_count * frm_shift_in_ms >= max_end_sil_frame_cnt_thresh) {
                    int lookback_frame = max_end_sil_frame_cnt_thresh / frm_shift_in_ms;
                    if (vad_opts.do_extend) {
                        lookback_frame -= vad_opts.lookahead_time_end_point / frm_shift_in_ms;
                        lookback_frame -= 1;
                        lookback_frame = std::max(0, lookback_frame);
                    }
                    OnVoiceEnd(cur_frm_idx - lookback_frame, false, false);
                    vad_state_machine = VadStateMachine::kVadInStateEndPointDetected;
                } else if (cur_frm_idx - confirmed_start_frame + 1 >
                           vad_opts.max_single_segment_time / frm_shift_in_ms) {
                    OnVoiceEnd(cur_frm_idx, false, false);
                    vad_state_machine = VadStateMachine::kVadInStateEndPointDetected;
                } else if (vad_opts.do_extend && !is_final_frame) {
                    if (continous_silence_frame_count <= vad_opts.lookahead_time_end_point / frm_shift_in_ms) {
                        OnVoiceDetected(cur_frm_idx);
                    }
                } else {
                    MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx);
                }
            }
        }
        if (vad_state_machine == VadStateMachine::kVadInStateEndPointDetected &&
            vad_opts.detect_mode == static_cast<int>(VadDetectMode::kVadMutipleUtteranceDetectMode)) {
            ResetDetection();
        }
    }

};



