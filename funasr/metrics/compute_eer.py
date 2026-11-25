import numpy as np
from sklearn.metrics import roc_curve
import argparse


def _compute_eer(label, pred, positive_label=1):
    """
    Python compute equal error rate (eer)
    ONLY tested on binary classification

    :param label: ground-truth label, should be a 1-d list or np.array, each element represents the ground-truth label of one sample
    :param pred: model prediction, should be a 1-d list or np.array, each element represents the model prediction of one sample
    :param positive_label: the class that is viewed as positive class when computing EER
    :return: equal error rate (EER)
    """

    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = roc_curve(label, pred, pos_label=positive_label)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer, eer_threshold


def compute_eer(trials_path, scores_path):
    """Compute eer.
    
        Args:
            trials_path: TODO.
            scores_path: TODO.
        """
    trials = {}
    with open(trials_path, "r") as trials_file:
        for line in trials_file:
            utt1, utt2, target = line.rstrip().split()
            trials[utt1 + " " + utt2] = target

    labels = []
    scores = []
    with open(scores_path, "r") as scores_file:
        for line in scores_file:
            utt1, utt2, score = line.rstrip().split()
            trial = utt1 + " " + utt2
            if trial not in trials:
                raise Exception("Missing entry for " + utt1 + " and " + utt2 + " " + str(scores_path))
            labels.append(trials[trial] == "target")
            scores.append(float(score))
    labels = np.array(labels, dtype=int)
    scores = np.array(scores, dtype=float)

    eer, threshold = _compute_eer(labels, scores)
    return eer, threshold


def main():
    """Main."""
    parser = argparse.ArgumentParser()
    parser.add_argument("trials", help="trial list")
    parser.add_argument("scores", help="score file, normalized to [0, 1]")
    args = parser.parse_args()

    eer, threshold = compute_eer(args.trials, args.scores)
    print("EER is {:.4f} at threshold {:.4f}".format(eer * 100.0, threshold))


if __name__ == "__main__":
    main()
