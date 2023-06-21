using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AliFsmnVadSharp.Model;

namespace AliFsmnVadSharp
{
    enum VadStateMachine
    {
        kVadInStateStartPointNotDetected = 1,
        kVadInStateInSpeechSegment = 2,
        kVadInStateEndPointDetected = 3,
    }
    enum VadDetectMode
    {
        kVadSingleUtteranceDetectMode = 0,
        kVadMutipleUtteranceDetectMode = 1,
    }


    internal class E2EVadModel
    {
        private VadPostConfEntity _vad_opts = new VadPostConfEntity();
        private WindowDetector _windows_detector = new WindowDetector();
        private bool _is_final = false;
        private int _data_buf_start_frame = 0;
        private int _frm_cnt = 0;
        private int _latest_confirmed_speech_frame = 0;
        private int _lastest_confirmed_silence_frame = -1;
        private int _continous_silence_frame_count = 0;
        private int _vad_state_machine = (int)VadStateMachine.kVadInStateStartPointNotDetected;
        private int _confirmed_start_frame = -1;
        private int _confirmed_end_frame = -1;
        private int _number_end_time_detected = 0;
        private int _sil_frame = 0;
        private int[] _sil_pdf_ids = new int[0];
        private double _noise_average_decibel = -100.0D;
        private bool _pre_end_silence_detected = false;
        private bool _next_seg = true;

        private List<E2EVadSpeechBufWithDoaEntity> _output_data_buf;
        private int _output_data_buf_offset = 0;
        private List<E2EVadFrameProbEntity> _frame_probs = new List<E2EVadFrameProbEntity>();
        private int _max_end_sil_frame_cnt_thresh = 800 - 150;
        private float _speech_noise_thres = 0.6F;
        private float[,,] _scores = null;
        private int _idx_pre_chunk = 0;
        private bool _max_time_out = false;
        private List<double> _decibel = new List<double>();
        private int _data_buf_size = 0;
        private int _data_buf_all_size = 0;

        public E2EVadModel(VadPostConfEntity vadPostConfEntity)
        {
            _vad_opts = vadPostConfEntity;
            _windows_detector = new WindowDetector(_vad_opts.window_size_ms,
                                                   _vad_opts.sil_to_speech_time_thres,
                                                   _vad_opts.speech_to_sil_time_thres,
                                                   _vad_opts.frame_in_ms);
            AllResetDetection();
        }

        private void AllResetDetection()
        {
            _is_final = false;
            _data_buf_start_frame = 0;
            _frm_cnt = 0;
            _latest_confirmed_speech_frame = 0;
            _lastest_confirmed_silence_frame = -1;
            _continous_silence_frame_count = 0;
            _vad_state_machine = (int)VadStateMachine.kVadInStateStartPointNotDetected;
            _confirmed_start_frame = -1;
            _confirmed_end_frame = -1;
            _number_end_time_detected = 0;
            _sil_frame = 0;
            _sil_pdf_ids = _vad_opts.sil_pdf_ids;
            _noise_average_decibel = -100.0F;
            _pre_end_silence_detected = false;
            _next_seg = true;

            _output_data_buf = new List<E2EVadSpeechBufWithDoaEntity>();
            _output_data_buf_offset = 0;
            _frame_probs = new List<E2EVadFrameProbEntity>();
            _max_end_sil_frame_cnt_thresh = _vad_opts.max_end_silence_time - _vad_opts.speech_to_sil_time_thres;
            _speech_noise_thres = _vad_opts.speech_noise_thres;
            _scores = null;
            _idx_pre_chunk = 0;
            _max_time_out = false;
            _decibel = new List<double>();
            _data_buf_size = 0;
            _data_buf_all_size = 0;
            ResetDetection();
        }

        private void ResetDetection()
        {
            _continous_silence_frame_count = 0;
            _latest_confirmed_speech_frame = 0;
            _lastest_confirmed_silence_frame = -1;
            _confirmed_start_frame = -1;
            _confirmed_end_frame = -1;
            _vad_state_machine = (int)VadStateMachine.kVadInStateStartPointNotDetected;
            _windows_detector.Reset();
            _sil_frame = 0;
            _frame_probs = new List<E2EVadFrameProbEntity>();
        }

        private void ComputeDecibel(float[] waveform)
        {
            int frame_sample_length = (int)(_vad_opts.frame_length_ms * _vad_opts.sample_rate / 1000);
            int frame_shift_length = (int)(_vad_opts.frame_in_ms * _vad_opts.sample_rate / 1000);
            if (_data_buf_all_size == 0)
            {
                _data_buf_all_size = waveform.Length;
                _data_buf_size = _data_buf_all_size;
            }
            else
            {
                _data_buf_all_size += waveform.Length;
            }

            for (int offset = 0; offset < waveform.Length - frame_sample_length + 1; offset += frame_shift_length)
            {
                float[] _waveform_chunk = new float[frame_sample_length];
                Array.Copy(waveform, offset, _waveform_chunk, 0, _waveform_chunk.Length);
                float[] _waveform_chunk_pow = _waveform_chunk.Select(x => (float)Math.Pow((double)x, 2)).ToArray();
                _decibel.Add(
                    10 * Math.Log10(
                        _waveform_chunk_pow.Sum() + 0.000001
                        )
                    );
            }

        }

        private void ComputeScores(float[,,] scores)
        {
            _vad_opts.nn_eval_block_size = scores.GetLength(1);
            _frm_cnt += scores.GetLength(1);
            _scores = scores;
        }

        private void PopDataBufTillFrame(int frame_idx)// need check again
        {
            while (_data_buf_start_frame < frame_idx)
            {
                if (_data_buf_size >= (int)(_vad_opts.frame_in_ms * _vad_opts.sample_rate / 1000))
                {
                    _data_buf_start_frame += 1;
                    _data_buf_size = _data_buf_all_size - _data_buf_start_frame * (int)(_vad_opts.frame_in_ms * _vad_opts.sample_rate / 1000);
                }
            }
        }

        private void PopDataToOutputBuf(int start_frm, int frm_cnt, bool first_frm_is_start_point,
                       bool last_frm_is_end_point, bool end_point_is_sent_end)
        {
            PopDataBufTillFrame(start_frm);
            int expected_sample_number = (int)(frm_cnt * _vad_opts.sample_rate * _vad_opts.frame_in_ms / 1000);
            if (last_frm_is_end_point)
            {
                int extra_sample = Math.Max(0, (int)(_vad_opts.frame_length_ms * _vad_opts.sample_rate / 1000 - _vad_opts.sample_rate * _vad_opts.frame_in_ms / 1000));
                expected_sample_number += (int)(extra_sample);
            }

            if (end_point_is_sent_end)
            {
                expected_sample_number = Math.Max(expected_sample_number, _data_buf_size);
            }
            if (_data_buf_size < expected_sample_number)
            {
                Console.WriteLine("error in calling pop data_buf\n");
            }

            if (_output_data_buf.Count == 0 || first_frm_is_start_point)
            {
                _output_data_buf.Add(new E2EVadSpeechBufWithDoaEntity());
                _output_data_buf.Last().Reset();
                _output_data_buf.Last().start_ms = start_frm * _vad_opts.frame_in_ms;
                _output_data_buf.Last().end_ms = _output_data_buf.Last().start_ms;
                _output_data_buf.Last().doa = 0;
            }

            E2EVadSpeechBufWithDoaEntity cur_seg = _output_data_buf.Last();
            if (cur_seg.end_ms != start_frm * _vad_opts.frame_in_ms)
            {
                Console.WriteLine("warning\n");
            }

            int out_pos = cur_seg.buffer.Length;  // cur_seg.buff现在没做任何操作
            int data_to_pop = 0;
            if (end_point_is_sent_end)
            {
                data_to_pop = expected_sample_number;
            }
            else
            {
                data_to_pop = (int)(frm_cnt * _vad_opts.frame_in_ms * _vad_opts.sample_rate / 1000);
            }
            if (data_to_pop > _data_buf_size)
            {
                Console.WriteLine("VAD data_to_pop is bigger than _data_buf_size!!!\n");
                data_to_pop = _data_buf_size;
                expected_sample_number = _data_buf_size;
            }


            cur_seg.doa = 0;
            for (int sample_cpy_out = 0; sample_cpy_out < data_to_pop; sample_cpy_out++)
            {
                out_pos += 1;
            }
            for (int sample_cpy_out = data_to_pop; sample_cpy_out < expected_sample_number; sample_cpy_out++)
            {
                out_pos += 1;
            }

            if (cur_seg.end_ms != start_frm * _vad_opts.frame_in_ms)
            {
                Console.WriteLine("Something wrong with the VAD algorithm\n");
            }

            _data_buf_start_frame += frm_cnt;
            cur_seg.end_ms = (start_frm + frm_cnt) * _vad_opts.frame_in_ms;
            if (first_frm_is_start_point)
            {
                cur_seg.contain_seg_start_point = true;
            }

            if (last_frm_is_end_point)
            {
                cur_seg.contain_seg_end_point = true;
            }
        }

        private void OnSilenceDetected(int valid_frame)
        {
            _lastest_confirmed_silence_frame = valid_frame;
            if (_vad_state_machine == (int)VadStateMachine.kVadInStateStartPointNotDetected)
            {
                PopDataBufTillFrame(valid_frame);
            }

        }

        private void OnVoiceDetected(int valid_frame)
        {
            _latest_confirmed_speech_frame = valid_frame;
            PopDataToOutputBuf(valid_frame, 1, false, false, false);
        }

        private void OnVoiceStart(int start_frame, bool fake_result = false)
        {
            if (_vad_opts.do_start_point_detection)
            {
                //do nothing
            }
            if (_confirmed_start_frame != -1)
            {

                Console.WriteLine("not reset vad properly\n");
            }
            else
            {
                _confirmed_start_frame = start_frame;
            }
            if (!fake_result || _vad_state_machine == (int)VadStateMachine.kVadInStateStartPointNotDetected)
            {

                PopDataToOutputBuf(_confirmed_start_frame, 1, true, false, false);
            }
        }

        private void OnVoiceEnd(int end_frame, bool fake_result, bool is_last_frame)
        {
            for (int t = _latest_confirmed_speech_frame + 1; t < end_frame; t++)
            {
                OnVoiceDetected(t);
            }
            if (_vad_opts.do_end_point_detection)
            {
                //do nothing
            }
            if (_confirmed_end_frame != -1)
            {
                Console.WriteLine("not reset vad properly\n");
            }
            else
            {
                _confirmed_end_frame = end_frame;
            }
            if (!fake_result)
            {
                _sil_frame = 0;
                PopDataToOutputBuf(_confirmed_end_frame, 1, false, true, is_last_frame);
            }
            _number_end_time_detected += 1;
        }

        private void MaybeOnVoiceEndIfLastFrame(bool is_final_frame, int cur_frm_idx)
        {
            if (is_final_frame)
            {
                OnVoiceEnd(cur_frm_idx, false, true);
                _vad_state_machine = (int)VadStateMachine.kVadInStateEndPointDetected;
            }
        }

        private int GetLatency()
        {
            return (int)(LatencyFrmNumAtStartPoint() * _vad_opts.frame_in_ms);
        }

        private int LatencyFrmNumAtStartPoint()
        {
            int vad_latency = _windows_detector.GetWinSize();
            if (_vad_opts.do_extend != 0)
            {
                vad_latency += (int)(_vad_opts.lookback_time_start_point / _vad_opts.frame_in_ms);
            }
            return vad_latency;
        }

        private FrameState GetFrameState(int t)
        {

            FrameState frame_state = FrameState.kFrameStateInvalid;
            double cur_decibel = _decibel[t];
            double cur_snr = cur_decibel - _noise_average_decibel;
            if (cur_decibel < _vad_opts.decibel_thres)
            {
                frame_state = FrameState.kFrameStateSil;
                DetectOneFrame(frame_state, t, false);
                return frame_state;
            }


            double sum_score = 0.0D;
            double noise_prob = 0.0D;
            Trace.Assert(_sil_pdf_ids.Length == _vad_opts.silence_pdf_num, "");
            if (_sil_pdf_ids.Length > 0)
            {
                Trace.Assert(_scores.GetLength(0) == 1, "只支持batch_size = 1的测试");  // 只支持batch_size = 1的测试
                float[] sil_pdf_scores = new float[_sil_pdf_ids.Length];
                int j = 0;
                foreach (int sil_pdf_id in _sil_pdf_ids)
                {
                    sil_pdf_scores[j] = _scores[0,t - _idx_pre_chunk,sil_pdf_id];
                    j++;
                }
                sum_score = sil_pdf_scores.Length == 0 ? 0 : sil_pdf_scores.Sum();
                noise_prob = Math.Log(sum_score) * _vad_opts.speech_2_noise_ratio;
                double total_score = 1.0D;
                sum_score = total_score - sum_score;
            }
            double speech_prob = Math.Log(sum_score);
            if (_vad_opts.output_frame_probs)
            {
                E2EVadFrameProbEntity frame_prob = new E2EVadFrameProbEntity();
                frame_prob.noise_prob = noise_prob;
                frame_prob.speech_prob = speech_prob;
                frame_prob.score = sum_score;
                frame_prob.frame_id = t;
                _frame_probs.Add(frame_prob);
            }

            if (Math.Exp(speech_prob) >= Math.Exp(noise_prob) + _speech_noise_thres)
            {
                if (cur_snr >= _vad_opts.snr_thres && cur_decibel >= _vad_opts.decibel_thres)
                {
                    frame_state = FrameState.kFrameStateSpeech;
                }
                else
                {
                    frame_state = FrameState.kFrameStateSil;
                }
            }
            else
            {
                frame_state = FrameState.kFrameStateSil;
                if (_noise_average_decibel < -99.9)
                {
                    _noise_average_decibel = cur_decibel;
                }
                else
                {
                    _noise_average_decibel = (cur_decibel + _noise_average_decibel * (_vad_opts.noise_frame_num_used_for_snr - 1)) / _vad_opts.noise_frame_num_used_for_snr;
                }
            }
            return frame_state;
        }

        public SegmentEntity[] DefaultCall(float[,,] score, float[] waveform,
            bool is_final = false, int max_end_sil = 800, bool online = false
            )
        {
            _max_end_sil_frame_cnt_thresh = max_end_sil - _vad_opts.speech_to_sil_time_thres;
            // compute decibel for each frame
            ComputeDecibel(waveform);
            ComputeScores(score);
            if (!is_final)
            {
                DetectCommonFrames();
            }
            else
            {
                DetectLastFrames();
            }
            int batchSize = score.GetLength(0);
            SegmentEntity[] segments = new SegmentEntity[batchSize];
            for (int batch_num = 0; batch_num < batchSize; batch_num++) // only support batch_size = 1 now
            {
                List<int[]> segment_batch = new List<int[]>();
                if (_output_data_buf.Count > 0)
                {
                    for (int i = _output_data_buf_offset; i < _output_data_buf.Count; i++)
                    {
                        int start_ms;
                        int end_ms;
                        if (online)
                        {
                            if (!_output_data_buf[i].contain_seg_start_point)
                            {
                                continue;
                            }
                            if (!_next_seg && !_output_data_buf[i].contain_seg_end_point)
                            {
                                continue;
                            }
                            start_ms = _next_seg ? _output_data_buf[i].start_ms : -1;
                            if (_output_data_buf[i].contain_seg_end_point)
                            {
                                end_ms = _output_data_buf[i].end_ms;
                                _next_seg = true;
                                _output_data_buf_offset += 1;
                            }
                            else
                            {
                                end_ms = -1;
                                _next_seg = false;
                            }
                        }
                        else
                        {
                            if (!is_final && (!_output_data_buf[i].contain_seg_start_point || !_output_data_buf[i].contain_seg_end_point))
                            {
                                continue;
                            }
                            start_ms = _output_data_buf[i].start_ms;
                            end_ms = _output_data_buf[i].end_ms;
                            _output_data_buf_offset += 1;

                        }
                        int[] segment_ms = new int[] { start_ms, end_ms };
                        segment_batch.Add(segment_ms);
                        
                    }

                }

                if (segment_batch.Count > 0)
                {
                    if (segments[batch_num] == null)
                    {
                        segments[batch_num] = new SegmentEntity();
                    }
                    segments[batch_num].Segment.AddRange(segment_batch);
                }
            }

            if (is_final)
            {
                // reset class variables and clear the dict for the next query
                AllResetDetection();
            }

            return segments;
        }

        private int DetectCommonFrames()
        {
            if (_vad_state_machine == (int)VadStateMachine.kVadInStateEndPointDetected)
            {
                return 0;
            }
            for (int i = _vad_opts.nn_eval_block_size - 1; i > -1; i += -1)
            {
                FrameState frame_state = FrameState.kFrameStateInvalid;
                frame_state = GetFrameState(_frm_cnt - 1 - i);
                DetectOneFrame(frame_state, _frm_cnt - 1 - i, false);
            }

            _idx_pre_chunk += _scores.GetLength(1)* _scores.GetLength(0); //_scores.shape[1];
            return 0;
        }

        private int DetectLastFrames()
        {
            if (_vad_state_machine == (int)VadStateMachine.kVadInStateEndPointDetected)
            {
                return 0;
            }
            for (int i = _vad_opts.nn_eval_block_size - 1; i > -1; i += -1)
            {
                FrameState frame_state = FrameState.kFrameStateInvalid;
                frame_state = GetFrameState(_frm_cnt - 1 - i);
                if (i != 0)
                {
                    DetectOneFrame(frame_state, _frm_cnt - 1 - i, false);
                }
                else
                {
                    DetectOneFrame(frame_state, _frm_cnt - 1, true);
                }


            }

            return 0;
        }

        private void DetectOneFrame(FrameState cur_frm_state, int cur_frm_idx, bool is_final_frame)
        {
            FrameState tmp_cur_frm_state = FrameState.kFrameStateInvalid;
            if (cur_frm_state == FrameState.kFrameStateSpeech)
            {
                if (Math.Abs(1.0) > _vad_opts.fe_prior_thres)//Fabs
                {
                    tmp_cur_frm_state = FrameState.kFrameStateSpeech;
                }
                else
                {
                    tmp_cur_frm_state = FrameState.kFrameStateSil;
                }
            }
            else if (cur_frm_state == FrameState.kFrameStateSil)
            {
                tmp_cur_frm_state = FrameState.kFrameStateSil;
            }

            AudioChangeState state_change = _windows_detector.DetectOneFrame(tmp_cur_frm_state, cur_frm_idx);
            int frm_shift_in_ms = _vad_opts.frame_in_ms;
            if (AudioChangeState.kChangeStateSil2Speech == state_change)
            {
                int silence_frame_count = _continous_silence_frame_count; // no used
                _continous_silence_frame_count = 0;
                _pre_end_silence_detected = false;
                int start_frame = 0;
                if (_vad_state_machine == (int)VadStateMachine.kVadInStateStartPointNotDetected)
                {
                    start_frame = Math.Max(_data_buf_start_frame, cur_frm_idx - LatencyFrmNumAtStartPoint());
                    OnVoiceStart(start_frame);
                    _vad_state_machine = (int)VadStateMachine.kVadInStateInSpeechSegment;
                    for (int t = start_frame + 1; t < cur_frm_idx + 1; t++)
                    {
                        OnVoiceDetected(t);
                    }

                }
                else if (_vad_state_machine == (int)VadStateMachine.kVadInStateInSpeechSegment)
                {
                    for (int t = _latest_confirmed_speech_frame + 1; t < cur_frm_idx; t++)
                    {
                        OnVoiceDetected(t);
                    }
                    if (cur_frm_idx - _confirmed_start_frame + 1 > _vad_opts.max_single_segment_time / frm_shift_in_ms)
                    {
                        OnVoiceEnd(cur_frm_idx, false, false);
                        _vad_state_machine = (int)VadStateMachine.kVadInStateEndPointDetected;
                    }

                    else if (!is_final_frame)
                    {
                        OnVoiceDetected(cur_frm_idx);
                    }
                    else
                    {
                        MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx);
                    }

                }
                else
                {
                    return;
                }
            }
            else if (AudioChangeState.kChangeStateSpeech2Sil == state_change)
            {
                _continous_silence_frame_count = 0;
                if (_vad_state_machine == (int)VadStateMachine.kVadInStateStartPointNotDetected)
                { return; }
                else if (_vad_state_machine == (int)VadStateMachine.kVadInStateInSpeechSegment)
                {
                    if (cur_frm_idx - _confirmed_start_frame + 1 > _vad_opts.max_single_segment_time / frm_shift_in_ms)
                    {
                        OnVoiceEnd(cur_frm_idx, false, false);
                        _vad_state_machine = (int)VadStateMachine.kVadInStateEndPointDetected;
                    }
                    else if (!is_final_frame)
                    {
                        OnVoiceDetected(cur_frm_idx);
                    }
                    else
                    {
                        MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx);
                    }

                }
                else
                {
                    return;
                }
            }
            else if (AudioChangeState.kChangeStateSpeech2Speech == state_change)
            {
                _continous_silence_frame_count = 0;
                if (_vad_state_machine == (int)VadStateMachine.kVadInStateInSpeechSegment)
                {
                    if (cur_frm_idx - _confirmed_start_frame + 1 > _vad_opts.max_single_segment_time / frm_shift_in_ms)
                    {
                        _max_time_out = true;
                        OnVoiceEnd(cur_frm_idx, false, false);
                        _vad_state_machine = (int)VadStateMachine.kVadInStateEndPointDetected;
                    }
                    else if (!is_final_frame)
                    {
                        OnVoiceDetected(cur_frm_idx);
                    }
                    else
                    {
                        MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx);
                    }
                }
                else
                {
                    return;
                }

            }
            else if (AudioChangeState.kChangeStateSil2Sil == state_change)
            {
                _continous_silence_frame_count += 1;
                if (_vad_state_machine == (int)VadStateMachine.kVadInStateStartPointNotDetected)
                {
                    // silence timeout, return zero length decision
                    if (((_vad_opts.detect_mode == (int)VadDetectMode.kVadSingleUtteranceDetectMode) && (
                            _continous_silence_frame_count * frm_shift_in_ms > _vad_opts.max_start_silence_time)) || (is_final_frame && _number_end_time_detected == 0))
                    {
                        for (int t = _lastest_confirmed_silence_frame + 1; t < cur_frm_idx; t++)
                        {
                            OnSilenceDetected(t);
                        }
                        OnVoiceStart(0, true);
                        OnVoiceEnd(0, true, false);
                        _vad_state_machine = (int)VadStateMachine.kVadInStateEndPointDetected;
                    }
                    else
                    {
                        if (cur_frm_idx >= LatencyFrmNumAtStartPoint())
                        {
                            OnSilenceDetected(cur_frm_idx - LatencyFrmNumAtStartPoint());
                        }
                    }
                }
                else if (_vad_state_machine == (int)VadStateMachine.kVadInStateInSpeechSegment)
                {
                    if (_continous_silence_frame_count * frm_shift_in_ms >= _max_end_sil_frame_cnt_thresh)
                    {
                        int lookback_frame = (int)(_max_end_sil_frame_cnt_thresh / frm_shift_in_ms);
                        if (_vad_opts.do_extend != 0)
                        {
                            lookback_frame -= (int)(_vad_opts.lookahead_time_end_point / frm_shift_in_ms);
                            lookback_frame -= 1;
                            lookback_frame = Math.Max(0, lookback_frame);
                        }

                        OnVoiceEnd(cur_frm_idx - lookback_frame, false, false);
                        _vad_state_machine = (int)VadStateMachine.kVadInStateEndPointDetected;
                    }
                    else if (cur_frm_idx - _confirmed_start_frame + 1 > _vad_opts.max_single_segment_time / frm_shift_in_ms)
                    {
                        OnVoiceEnd(cur_frm_idx, false, false);
                        _vad_state_machine = (int)VadStateMachine.kVadInStateEndPointDetected;
                    }

                    else if (_vad_opts.do_extend != 0 && !is_final_frame)
                    {
                        if (_continous_silence_frame_count <= (int)(_vad_opts.lookahead_time_end_point / frm_shift_in_ms))
                        {
                            OnVoiceDetected(cur_frm_idx);
                        }
                    }

                    else
                    {
                        MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx);
                    }
                }
                else
                {
                    return;
                }

            }

            if (_vad_state_machine == (int)VadStateMachine.kVadInStateEndPointDetected && _vad_opts.detect_mode == (int)VadDetectMode.kVadMutipleUtteranceDetectMode)
            {
                ResetDetection();
            }

        }

    }
}
