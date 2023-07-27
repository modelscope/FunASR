// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AliFsmnVad
{
    public enum FrameState
    {
        kFrameStateInvalid = -1,
        kFrameStateSpeech = 1,
        kFrameStateSil = 0
    }

    /// <summary>
    /// final voice/unvoice state per frame
    /// </summary>
    public enum AudioChangeState
    {
        kChangeStateSpeech2Speech = 0,
        kChangeStateSpeech2Sil = 1,
        kChangeStateSil2Sil = 2,
        kChangeStateSil2Speech = 3,
        kChangeStateNoBegin = 4,
        kChangeStateInvalid = 5
    }


    internal class WindowDetector
    {
        private int _window_size_ms = 0; 
        private int _sil_to_speech_time = 0;
        private int _speech_to_sil_time = 0;
        private int _frame_size_ms = 0;

        private int _win_size_frame = 0;
        private int _win_sum = 0;
        private int[] _win_state = new int[0];  // 初始化窗

        private int _cur_win_pos = 0;
        private int _pre_frame_state = (int)FrameState.kFrameStateSil;
        private int _cur_frame_state = (int)FrameState.kFrameStateSil;
        private int _sil_to_speech_frmcnt_thres = 0; //int(sil_to_speech_time / frame_size_ms);
        private int _speech_to_sil_frmcnt_thres = 0; //int(speech_to_sil_time / frame_size_ms);

        private int _voice_last_frame_count = 0;
        private int _noise_last_frame_count = 0;
        private int _hydre_frame_count = 0;

        public WindowDetector()
        {

        }

        public WindowDetector(int window_size_ms, int sil_to_speech_time, int speech_to_sil_time, int frame_size_ms)
        {
            _window_size_ms = window_size_ms;
            _sil_to_speech_time = sil_to_speech_time;
            _speech_to_sil_time = speech_to_sil_time;
            _frame_size_ms = frame_size_ms;

            _win_size_frame = (int)(window_size_ms / frame_size_ms);
            _win_sum = 0;
            _win_state = new int[_win_size_frame];  // 初始化窗

            _cur_win_pos = 0;
            _pre_frame_state = (int)FrameState.kFrameStateSil;
            _cur_frame_state = (int)FrameState.kFrameStateSil;
            _sil_to_speech_frmcnt_thres = (int)(sil_to_speech_time / frame_size_ms);
            _speech_to_sil_frmcnt_thres = (int)(speech_to_sil_time / frame_size_ms);

            _voice_last_frame_count = 0;
            _noise_last_frame_count = 0;
            _hydre_frame_count = 0;
        }

        public void Reset()
        {
            _cur_win_pos = 0;
            _win_sum = 0;
            _win_state = new int[_win_size_frame];
            _pre_frame_state = (int)FrameState.kFrameStateSil;
            _cur_frame_state = (int)FrameState.kFrameStateSil;
            _voice_last_frame_count = 0;
            _noise_last_frame_count = 0;
            _hydre_frame_count = 0;
        }
        

        public int GetWinSize()
        {
            return _win_size_frame;
        }

        public AudioChangeState DetectOneFrame(FrameState frameState, int frame_count)
        {


            _cur_frame_state = (int)FrameState.kFrameStateSil;
            if (frameState == FrameState.kFrameStateSpeech)
            {
                _cur_frame_state = 1;
            }

            else if (frameState == FrameState.kFrameStateSil)
            {
                _cur_frame_state = 0;
            }

            else
            {
                return AudioChangeState.kChangeStateInvalid;
            }

            _win_sum -= _win_state[_cur_win_pos];
            _win_sum += _cur_frame_state;
            _win_state[_cur_win_pos] = _cur_frame_state;
            _cur_win_pos = (_cur_win_pos + 1) % _win_size_frame;

            if (_pre_frame_state == (int)FrameState.kFrameStateSil && _win_sum >= _sil_to_speech_frmcnt_thres)
            {
                _pre_frame_state = (int)FrameState.kFrameStateSpeech;
                return AudioChangeState.kChangeStateSil2Speech;
            }


            if (_pre_frame_state == (int)FrameState.kFrameStateSpeech && _win_sum <= _speech_to_sil_frmcnt_thres)
            {
                _pre_frame_state = (int)FrameState.kFrameStateSil;
                return AudioChangeState.kChangeStateSpeech2Sil;
            }


            if (_pre_frame_state == (int)FrameState.kFrameStateSil)
            {
                return AudioChangeState.kChangeStateSil2Sil;
            }

            if (_pre_frame_state == (int)FrameState.kFrameStateSpeech)
            {
                return AudioChangeState.kChangeStateSpeech2Speech;
            }

            return AudioChangeState.kChangeStateInvalid;
        }

        private int FrameSizeMs()
        {
            return _frame_size_ms;
        }



    }
}
