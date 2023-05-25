# Copyright 2021  NPU, ASLP Group (Author: Qijie Shao)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# process_opus.py: segmentation and downsampling of opus audio

# usage: python3 process_opus.py wav.scp segments output_wav.scp

from pydub import AudioSegment
import sys
import os


def read_file(wav_scp, segments):
    wav_scp_dict = {}
    with open(wav_scp, 'r', encoding='UTF-8') as fin:
        for line_str in fin:
            wav_id, path = line_str.strip().split()
            wav_scp_dict[wav_id] = path

    utt_list = []
    seg_path_list = []
    start_time_list = []
    end_time_list = []
    with open(segments, 'r', encoding='UTF-8') as fin:
        for line_str in fin:
            arr = line_str.strip().split()
            assert len(arr) == 4
            utt_list.append(arr[0])
            seg_path_list.append(wav_scp_dict[arr[1]])
            start_time_list.append(float(arr[2]))
            end_time_list.append(float(arr[3]))
    return utt_list, seg_path_list, start_time_list, end_time_list


# TODO(Qijie): Fix the process logic
def output(output_wav_scp, utt_list, seg_path_list, start_time_list,
           end_time_list):
    num_utts = len(utt_list)
    step = int(num_utts * 0.01)
    with open(output_wav_scp, 'w', encoding='UTF-8') as fout:
        previous_wav_path = ""
        for i in range(num_utts):
            utt_id = utt_list[i]
            current_wav_path = seg_path_list[i]
            output_dir = (os.path.dirname(current_wav_path)) \
                .replace("audio", 'audio_seg')
            seg_wav_path = os.path.join(output_dir, utt_id + '.wav')

            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir)

            if current_wav_path != previous_wav_path:
                source_wav = AudioSegment.from_file(current_wav_path)
            previous_wav_path = current_wav_path

            start = int(start_time_list[i] * 1000)
            end = int(end_time_list[i] * 1000)
            target_audio = source_wav[start:end].set_frame_rate(16000) \
                .set_sample_width(2)
            target_audio.export(seg_wav_path, format="wav")

            fout.write("{} {}\n".format(utt_id, seg_wav_path))
            if i % step == 0:
                print("seg wav finished: {}%".format(int(i / step)))


def main():
    wav_scp = sys.argv[1]
    segments = sys.argv[2]
    output_wav_scp = sys.argv[3]

    utt_list, seg_path_list, start_time_list, end_time_list \
        = read_file(wav_scp, segments)
    output(output_wav_scp, utt_list, seg_path_list, start_time_list,
           end_time_list)


if __name__ == '__main__':
    main()
