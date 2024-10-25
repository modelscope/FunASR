from argparse import ArgumentParser

from fun_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
from fun_text_processing.text_normalization.data_loader_utils import (
    evaluate,
    known_types,
    load_files,
    training_data_to_sentences,
    training_data_to_tokens,
)

"""
Runs Evaluation on data in the format of : <semiotic class>\t<unnormalized text>\t<`self` if trivial class or normalized text>
like the Google text normalization data https://www.kaggle.com/richardwilliamsproat/text-normalization-for-english-russian-and-polish
"""

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input", help="input file path", type=str, required=True)
    parser.add_argument(
        "--lang",
        help="language",
        choices=["en", "id", "ja", "de", "es", "pt", "ru", "fr", "vi", "ko", "zh", "fil"],
        default="en",
        type=str,
    )
    parser.add_argument(
        "--cat",
        dest="category",
        help="focus on class only (" + ", ".join(known_types) + ")",
        type=str,
        default=None,
        choices=known_types,
    )
    parser.add_argument(
        "--filter", action="store_true", help="clean data for inverse normalization purposes"
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Example usage:
    # python run_evaluate.py --input=<INPUT> --cat=<CATEGORY> --filter
    args = parse_args()
    if args.lang == "en":
        from fun_text_processing.inverse_text_normalization.en.clean_eval_data import filter_loaded_data

    file_path = args.input
    inverse_normalizer = InverseNormalizer()

    print("Loading training data: " + file_path)
    training_data = load_files([file_path])

    if args.filter:
        training_data = filter_loaded_data(training_data)

    # Evaluate at sentence level if no specific category is provided
    if args.category is None:
        print("Sentence level evaluation...")
        sentences_un_normalized, sentences_normalized, _ = training_data_to_sentences(training_data)
        print("- Data: " + str(len(sentences_normalized)) + " sentences")
        sentences_prediction = inverse_normalizer.inverse_normalize_list(sentences_normalized)
        print("- Denormalized. Evaluating...")
        sentences_accuracy = evaluate(
            preds=sentences_prediction, labels=sentences_un_normalized, input=sentences_normalized
        )
        print("- Accuracy: " + str(sentences_accuracy))

    # Evaluate at token level
    print("Token level evaluation...")
    tokens_per_type = training_data_to_tokens(training_data, category=args.category)
    token_accuracy = {}
    for token_type, (tokens_un_normalized, tokens_normalized) in tokens_per_type.items():
        print("- Token type: " + token_type)
        print("  - Data: " + str(len(tokens_normalized)) + " tokens")
        tokens_prediction = inverse_normalizer.inverse_normalize_list(tokens_normalized)
        print("  - Denormalized. Evaluating...")
        token_accuracy[token_type] = evaluate(
            tokens_prediction, tokens_un_normalized, input=tokens_normalized
        )
        print("  - Accuracy: " + str(token_accuracy[token_type]))

    # Calculate weighted token accuracy
    token_count_per_type = {token_type: len(tokens) for token_type, (tokens, _) in tokens_per_type.items()}
    token_weighted_accuracy = [
        token_count_per_type[token_type] * accuracy
        for token_type, accuracy in token_accuracy.items()
    ]
    print("- Accuracy: " + str(sum(token_weighted_accuracy) / sum(token_count_per_type.values())))

    print(" - Total: " + str(sum(token_count_per_type.values())), "\n")

    for token_type in token_accuracy:
        if token_type not in known_types:
            raise ValueError("Unexpected token type: " + token_type)

    # Output table summarizing evaluation results if no specific category is provided
    if args.category is None:
        c1 = ["Class", "sent level"] + known_types
        c2 = ["Num Tokens", len(sentences_normalized)] + [
            str(token_count_per_type.get(known_type, 0)) for known_type in known_types
        ]
        c3 = ["Denormalization", str(sentences_accuracy)] + [
            str(token_accuracy.get(known_type, "0")) for known_type in known_types
        ]
        for i in range(len(c1)):
            print(f"{c1[i]:10s} | {c2[i]:10s} | {c3[i]:5s}")
    else:
        print(f"numbers\t{token_count_per_type[args.category]}")
        print(f"Denormalization\t{token_accuracy[args.category]}")
