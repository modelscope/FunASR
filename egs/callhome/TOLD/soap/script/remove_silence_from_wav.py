import numpy as np
from funasr.utils.job_runner import MultiProcessRunnerV3
from funasr.utils.misc import load_scp_as_list, load_scp_as_dict
import os
import librosa
import soundfile as sf


class MyRunner(MultiProcessRunnerV3):

    def prepare(self, parser):
        parser.add_argument("dir", type=str)
        parser.add_argument("out_dir", type=str)
        args = parser.parse_args()
        assert args.sr == 8000, "For callhome dataset, the sample rate should be 8000, use --sr 8000."

        meeting_scp = load_scp_as_list(os.path.join(args.dir, "reco.scp"))
        vad_file = open(os.path.join(args.dir, "segments"))
        meeting2vad = {}
        for one_line in vad_file:
            uid, mid, st, ed = one_line.strip().split(" ")
            st, ed = int(float(st) * args.sr), int(float(ed) * args.sr)
            if mid not in meeting2vad:
                meeting2vad[mid] = []
            meeting2vad[mid].append((uid, st, ed))

        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

        for mid, _ in meeting_scp:
            if mid not in meeting2vad:
                print("Recording {} doesn't contains speech segments".format(mid))
        task_list = [(mid, wav_path, meeting2vad[mid]) for mid, wav_path in meeting_scp if mid in meeting2vad]
        return task_list, None, args

    def post(self, results_list, args):
        pass


def process(task_args):
    _, task_list, _, args = task_args
    for mid, wav_path, vad_list in task_list:
        wav = librosa.load(wav_path, args.sr, True)[0] * 32767
        seg_list = []
        pos_map = []
        offset = 0
        for uid, st, ed in vad_list:
            seg_list.append(wav[st: ed])
            pos_map.append("{} {} {} {} {}\n".format(uid, st, ed, offset, offset+ed-st))
            offset = offset + ed - st
        out = np.concatenate(seg_list, axis=0)
        save_path = os.path.join(args.out_dir, "{}.wav".format(mid))
        sf.write(save_path, out.astype(np.int16), args.sr, "PCM_16", "LITTLE", "WAV", True)
        map_path = os.path.join(args.out_dir, "{}.pos".format(mid))
        with open(map_path, "wt") as fd:
            fd.writelines(pos_map)
        # print mid
    return None


if __name__ == '__main__':
    my_runner = MyRunner(process)
    my_runner.run()
