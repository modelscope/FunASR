def split_to_mini_sentence(words: list, word_limit: int = 20):
    assert word_limit > 1
    if len(words) <= word_limit:
        return [words]
    sentences = []
    length = len(words)
    sentence_len = length // word_limit
    for i in range(sentence_len):
        sentences.append(words[i * word_limit:(i + 1) * word_limit])
    if length % word_limit > 0:
        sentences.append(words[sentence_len * word_limit:])
    return sentences
