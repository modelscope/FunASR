#!/usr/bin/env python3
# Copyright 2018  David Snyder
# Apache 2.0

# This script computes the minimum detection cost function, which is a common
# error metric used in speaker recognition.  Compared to equal error-rate,
# which assigns equal weight to false negatives and false positives, this
# error-rate is usually used to assess performance in settings where achieving
# a low false positive rate is more important than achieving a low false
# negative rate.  See the NIST 2016 Speaker Recognition Evaluation Plan at
# https://www.nist.gov/sites/default/files/documents/2016/10/07/sre16_eval_plan_v1.3.pdf
# for more details about the metric.
from __future__ import print_function
from operator import itemgetter
import sys, argparse, os


def GetArgs():
    parser = argparse.ArgumentParser(description="Compute the minimum "
                                                 "detection cost function along with the threshold at which it occurs. "
                                                 "Usage: sid/compute_min_dcf.py [options...] <scores-file> "
                                                 "<trials-file> "
                                                 "E.g., sid/compute_min_dcf.py --p-target 0.01 --c-miss 1 --c-fa 1 "
                                                 "exp/scores/trials data/test/trials",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--p-target', type=float, dest="p_target",
                        default=0.01,
                        help='The prior probability of the target speaker in a trial.')
    parser.add_argument('--c-miss', type=float, dest="c_miss", default=1,
                        help='Cost of a missed detection.  This is usually not changed.')
    parser.add_argument('--c-fa', type=float, dest="c_fa", default=1,
                        help='Cost of a spurious detection.  This is usually not changed.')
    parser.add_argument("scores_filename",
                        help="Input scores file, with columns of the form "
                             "<utt1> <utt2> <score>")
    parser.add_argument("trials_filename",
                        help="Input trials file, with columns of the form "
                             "<utt1> <utt2> <target/nontarget>")
    sys.stderr.write(' '.join(sys.argv) + "\n")
    args = parser.parse_args()
    args = CheckArgs(args)
    return args


def CheckArgs(args):
    if args.c_fa <= 0:
      raise Exception("--c-fa must be greater than 0")
    if args.c_miss <= 0:
      raise Exception("--c-miss must be greater than 0")
    if args.p_target <= 0 or args.p_target >= 1:
      raise Exception("--p-target must be greater than 0 and less than 1")
    return args


# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):

    # Sort the scores from smallest to largest, and also get the corresponding
    # indexes of the sorted scores.  We will treat the sorted scores as the
    # thresholds at which the the error-rates are evaluated.
    sorted_indexes, thresholds = zip(*sorted(
        [(index, threshold) for index, threshold in enumerate(scores)],
        key=itemgetter(1)))
    labels = [labels[i] for i in sorted_indexes]
    fns = []
    tns = []

    # At the end of this loop, fns[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i]. And, tns[i]
    # is the total number of times that we have correctly rejected scores
    # less than thresholds[i].
    for i in range(0, len(labels)):
        if i == 0:
            fns.append(labels[i])
            tns.append(1 - labels[i])
        else:
            fns.append(fns[i-1] + labels[i])
            tns.append(tns[i-1] + 1 - labels[i])
    positives = sum(labels)
    negatives = len(labels) - positives

    # Now divide the false negatives by the total number of 
    # positives to obtain the false negative rates across
    # all thresholds
    fnrs = [fn / float(positives) for fn in fns]

    # Divide the true negatives by the total number of 
    # negatives to get the true negative rate. Subtract these 
    # quantities from 1 to get the false positive rates.
    fprs = [1 - tn / float(negatives) for tn in tns]
    return fnrs, fprs, thresholds


# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold


def compute_min_dcf(scores_filename, trials_filename, c_miss=1, c_fa=1, p_target=0.01):
    scores_file = open(scores_filename, 'r').readlines()
    trials_file = open(trials_filename, 'r').readlines()
    c_miss = c_miss
    c_fa = c_fa
    p_target = p_target

    scores = []
    labels = []

    trials = {}
    for line in trials_file:
        utt1, utt2, target = line.rstrip().split()
        trial = utt1 + " " + utt2
        trials[trial] = target

    for line in scores_file:
        utt1, utt2, score = line.rstrip().split()
        trial = utt1 + " " + utt2
        if trial in trials:
            scores.append(float(score))
            if trials[trial] == "target":
                labels.append(1)
            else:
                labels.append(0)
        else:
            raise Exception("Missing entry for " + utt1 + " and " + utt2
                            + " " + scores_filename)

    fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
    mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, p_target,
                                      c_miss, c_fa)
    return mindcf, threshold


def main():
    args = GetArgs()
    mindcf, threshold = compute_min_dcf(
        args.scores_filename, args.trials_filename,
        args.c_miss, args.c_fa, args.p_target
    )
    sys.stdout.write("minDCF is {0:.4f} at threshold {1:.4f} (p-target={2}, c-miss={3}, "
                     "c-fa={4})\n".format(mindcf, threshold, args.p_target, args.c_miss, args.c_fa))


if __name__ == "__main__":
    main()
