import random


def sample_hotword(
    length,
    hotword_min_length,
    hotword_max_length,
    sample_rate,
    double_rate,
    pre_prob,
    pre_index=None,
    pre_hwlist=None,
):
    if length < hotword_min_length:
        return [-1]
    if random.random() < sample_rate:
        if pre_prob > 0 and random.random() < pre_prob and pre_index is not None:
            return pre_index
        if length == hotword_min_length:
            return [0, length - 1]
        elif random.random() < double_rate and length > hotword_max_length + hotword_min_length + 2:
            # sample two hotwords in a sentence
            _max_hw_length = min(hotword_max_length, length // 2)
            # first hotword
            start1 = random.randint(0, length // 3)
            end1 = random.randint(start1 + hotword_min_length - 1, start1 + _max_hw_length - 1)
            # second hotword
            start2 = random.randint(end1 + 1, length - hotword_min_length)
            end2 = random.randint(
                min(length - 1, start2 + hotword_min_length - 1),
                min(length - 1, start2 + hotword_max_length - 1),
            )
            return [start1, end1, start2, end2]
        else:  # single hotword
            start = random.randint(0, length - hotword_min_length)
            end = random.randint(
                min(length - 1, start + hotword_min_length - 1),
                min(length - 1, start + hotword_max_length - 1),
            )
            return [start, end]
    else:
        return [-1]
