// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AliFsmnVad.Model
{
    public class VadPostConfEntity
    {
        private int _sample_rate= 16000;
        private int _detect_mode = 1 ;
        private int _snr_mode = 0;
        private int _max_end_silence_time = 800;
        private int _max_start_silence_time = 3000;
        private bool _do_start_point_detection = true;
        private bool _do_end_point_detection = true;
        private int _window_size_ms = 200;
        private int _sil_to_speech_time_thres = 150;
        private int _speech_to_sil_time_thres = 150;
        private float _speech_2_noise_ratio = 1.0F;
        private int _do_extend = 1;
        private int _lookback_time_start_point = 200;
        private int _lookahead_time_end_point = 100;
        private int _max_single_segment_time = 60000;
        private int _nn_eval_block_size = 8;
        private int _dcd_block_size = 4;
        private float _snr_thres = -100.0F;
        private int _noise_frame_num_used_for_snr = 100;
        private float _decibel_thres = -100.0F;
        private float _speech_noise_thres = 0.6F;
        private float _fe_prior_thres = 0.0001F;
        private int _silence_pdf_num = 1;
        private int[] _sil_pdf_ids = new int[] {0};
        private float _speech_noise_thresh_low = -0.1F;
        private float _speech_noise_thresh_high = 0.3F;
        private bool _output_frame_probs = false;
        private int _frame_in_ms = 10;
        private int _frame_length_ms = 25;

        public int sample_rate { get => _sample_rate; set => _sample_rate = value; }
        public int detect_mode { get => _detect_mode; set => _detect_mode = value; }
        public int snr_mode { get => _snr_mode; set => _snr_mode = value; }
        public int max_end_silence_time { get => _max_end_silence_time; set => _max_end_silence_time = value; }
        public int max_start_silence_time { get => _max_start_silence_time; set => _max_start_silence_time = value; }
        public bool do_start_point_detection { get => _do_start_point_detection; set => _do_start_point_detection = value; }
        public bool do_end_point_detection { get => _do_end_point_detection; set => _do_end_point_detection = value; }
        public int window_size_ms { get => _window_size_ms; set => _window_size_ms = value; }
        public int sil_to_speech_time_thres { get => _sil_to_speech_time_thres; set => _sil_to_speech_time_thres = value; }
        public int speech_to_sil_time_thres { get => _speech_to_sil_time_thres; set => _speech_to_sil_time_thres = value; }
        public float speech_2_noise_ratio { get => _speech_2_noise_ratio; set => _speech_2_noise_ratio = value; }
        public int do_extend { get => _do_extend; set => _do_extend = value; }
        public int lookback_time_start_point { get => _lookback_time_start_point; set => _lookback_time_start_point = value; }
        public int lookahead_time_end_point { get => _lookahead_time_end_point; set => _lookahead_time_end_point = value; }
        public int max_single_segment_time { get => _max_single_segment_time; set => _max_single_segment_time = value; }
        public int nn_eval_block_size { get => _nn_eval_block_size; set => _nn_eval_block_size = value; }
        public int dcd_block_size { get => _dcd_block_size; set => _dcd_block_size = value; }
        public float snr_thres { get => _snr_thres; set => _snr_thres = value; }
        public int noise_frame_num_used_for_snr { get => _noise_frame_num_used_for_snr; set => _noise_frame_num_used_for_snr = value; }
        public float decibel_thres { get => _decibel_thres; set => _decibel_thres = value; }
        public float speech_noise_thres { get => _speech_noise_thres; set => _speech_noise_thres = value; }
        public float fe_prior_thres { get => _fe_prior_thres; set => _fe_prior_thres = value; }
        public int silence_pdf_num { get => _silence_pdf_num; set => _silence_pdf_num = value; }
        public int[] sil_pdf_ids { get => _sil_pdf_ids; set => _sil_pdf_ids = value; }
        public float speech_noise_thresh_low { get => _speech_noise_thresh_low; set => _speech_noise_thresh_low = value; }
        public float speech_noise_thresh_high { get => _speech_noise_thresh_high; set => _speech_noise_thresh_high = value; }
        public bool output_frame_probs { get => _output_frame_probs; set => _output_frame_probs = value; }
        public int frame_in_ms { get => _frame_in_ms; set => _frame_in_ms = value; }
        public int frame_length_ms { get => _frame_length_ms; set => _frame_length_ms = value; }
        
    }
}
