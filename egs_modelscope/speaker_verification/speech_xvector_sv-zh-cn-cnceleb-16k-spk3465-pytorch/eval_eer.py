from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, default="eval")
    parser.add_argument("--trials", type=str, default="eval/lists/trials.lst.speech")
    parser.add_argument("--out_dir", type=str, default="./")
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    inference_sv_pipline = pipeline(
        task=Tasks.speaker_verification,
        model='damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch'
    )

    trials_list = [x.strip() for x in open(args.trials, "r").readlines()]
    enroll_list = set([x.split(" ")[0] for x in trials_list])
    test_list = set([x.split(" ")[1] for x in trials_list])

    print("extract embeddings for {} enrollments".format(len(enroll_list)))
    enroll_embedding = {}
    for enroll in enroll_list:
        spk_embedding = inference_sv_pipline(
            audio_in=os.path.join(args.eval_dir, "enroll", enroll+".wav")
        )["spk_embedding"]
        enroll_embedding[enroll] = spk_embedding

    test_embedding = {}
    print("extract embeddings for {} tests".format(len(test_list)))
    for test in test_list:
        spk_embedding = inference_sv_pipline(
            audio_in=os.path.join(args.eval_dir, "test", test+".wav")
        )["spk_embedding"]
        test_embedding[test] = spk_embedding

    print("calculate scores for {} trials".format(len(trials_list)))
    fd = open(os.path.join(args.out_dir, "scores"), "w")
    for trial in trials_list:
        spk, utt, _ = trial.split(" ")
        spk_emb = enroll_embedding[spk]
        utt_emb = test_embedding[utt]
        score = np.sum(spk_emb * utt_emb) / (np.linalg.norm(spk_emb) * np.linalg.norm(utt_emb))
        fd.write("{} {} {:.5f}\n".format(spk, utt, score))
    fd.close()

    from funasr.utils.compute_eer import compute_eer
    from funasr.utils.compute_min_dcf import compute_min_dcf
    eer, threshold = compute_eer(args.trials, os.path.join(args.out_dir, "scores"))
    print("EER is {:.4f} at threshold {:.4f}".format(eer * 100.0, threshold))

    mindcf, threshold = compute_min_dcf(
        os.path.join(args.out_dir, "scores"), args.trials,
        c_miss=10, p_target=0.01
    )
    print("minDCF is {0:.4f} at threshold {1:.4f} (p-target={2}, c-miss={3}, c-fa={4})\n".format(
        mindcf, threshold, 0.01, 10, 1
    ))


if __name__ == '__main__':
    main()
