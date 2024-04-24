import json
import os
import time
from argparse import ArgumentParser
from glob import glob
from typing import List, Tuple

import pynini
from joblib import Parallel, delayed
from fun_text_processing.text_normalization.data_loader_utils import post_process_punct, pre_process
from fun_text_processing.text_normalization.normalize import Normalizer
from pynini.lib import rewrite
from tqdm import tqdm

try:
    from nemo.collections.asr.metrics.wer import word_error_rate
    from nemo.collections.asr.models import ASRModel

    ASR_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    ASR_AVAILABLE = False


"""
The script provides multiple normalization options and chooses the best one that minimizes CER of the ASR output
(most of the semiotic classes use deterministic=False flag).

To run this script with a .json manifest file, the manifest file should contain the following fields:
    "audio_data" - path to the audio file
    "text" - raw text
    "pred_text" - ASR model prediction

    See https://github.com/NVIDIA/NeMo/blob/main/examples/asr/transcribe_speech.py on how to add ASR predictions

    When the manifest is ready, run:
        python normalize_with_audio.py \
               --audio_data PATH/TO/MANIFEST.JSON \
               --language en


To run with a single audio file, specify path to audio and text with:
    python normalize_with_audio.py \
           --audio_data PATH/TO/AUDIO.WAV \
           --language en \
           --text raw text OR PATH/TO/.TXT/FILE
           --model QuartzNet15x5Base-En \
           --verbose

To see possible normalization options for a text input without an audio file (could be used for debugging), run:
    python python normalize_with_audio.py --text "RAW TEXT"

Specify `--cache_dir` to generate .far grammars once and re-used them for faster inference
"""


class NormalizerWithAudio(Normalizer):
    """
    Normalizer class that converts text from written to spoken form.
    Useful for TTS preprocessing.

    Args:
        input_case: expected input capitalization
        lang: language
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
        whitelist: path to a file with whitelist replacements
        post_process: WFST-based post processing, e.g. to remove extra spaces added during TN.
            Note: punct_post_process flag in normalize() supports all languages.
    """

    def __init__(
        self,
        input_case: str,
        lang: str = "en",
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
        lm: bool = False,
        post_process: bool = True,
    ):

        super().__init__(
            input_case=input_case,
            lang=lang,
            deterministic=False,
            cache_dir=cache_dir,
            overwrite_cache=overwrite_cache,
            whitelist=whitelist,
            lm=lm,
            post_process=post_process,
        )
        self.lm = lm

    def normalize(
        self,
        text: str,
        n_tagged: int,
        punct_post_process: bool = True,
        verbose: bool = False,
    ) -> str:
        """
        Main function. Normalizes tokens from written to spoken form
            e.g. 12 kg -> twelve kilograms

        Args:
            text: string that may include semiotic classes
            n_tagged: number of tagged options to consider, -1 - to get all possible tagged options
            punct_post_process: whether to normalize punctuation
            verbose: whether to print intermediate meta information

        Returns:
            normalized text options (usually there are multiple ways of normalizing a given semiotic class)
        """

        if len(text.split()) > 500:
            raise ValueError(
                "Your input is too long. Please split up the input into sentences, "
                "or strings with fewer than 500 words"
            )

        original_text = text
        text = pre_process(text)  # to handle []

        text = text.strip()
        if not text:
            if verbose:
                print(text)
            return text
        text = pynini.escape(text)
        print(text)

        if self.lm:
            if self.lang not in ["en"]:
                raise ValueError(f"{self.lang} is not supported in LM mode")

            if self.lang == "en":
                # this to keep arpabet phonemes in the list of options
                if "[" in text and "]" in text:

                    lattice = rewrite.rewrite_lattice(text, self.tagger.fst)
                else:
                    try:
                        lattice = rewrite.rewrite_lattice(text, self.tagger.fst_no_digits)
                    except pynini.lib.rewrite.Error:
                        lattice = rewrite.rewrite_lattice(text, self.tagger.fst)
                lattice = rewrite.lattice_to_nshortest(lattice, n_tagged)
                tagged_texts = [(x[1], float(x[2])) for x in lattice.paths().items()]
                tagged_texts.sort(key=lambda x: x[1])
                tagged_texts, weights = list(zip(*tagged_texts))
        else:
            tagged_texts = self._get_tagged_text(text, n_tagged)
        # non-deterministic Eng normalization uses tagger composed with verbalizer, no permutation in between
        if self.lang == "en":
            normalized_texts = tagged_texts
            normalized_texts = [self.post_process(text) for text in normalized_texts]
        else:
            normalized_texts = []
            for tagged_text in tagged_texts:
                self._verbalize(tagged_text, normalized_texts, verbose=verbose)

        if len(normalized_texts) == 0:
            raise ValueError()

        if punct_post_process:
            # do post-processing based on Moses detokenizer
            if self.processor:
                normalized_texts = [self.processor.detokenize([t]) for t in normalized_texts]
                normalized_texts = [
                    post_process_punct(input=original_text, normalized_text=t)
                    for t in normalized_texts
                ]

        if self.lm:
            remove_dup = sorted(list(set(zip(normalized_texts, weights))), key=lambda x: x[1])
            normalized_texts, weights = zip(*remove_dup)
            return list(normalized_texts), weights

        normalized_texts = set(normalized_texts)
        return normalized_texts

    def _get_tagged_text(self, text, n_tagged):
        """
        Returns text after tokenize and classify
        Args;
            text: input  text
            n_tagged: number of tagged options to consider, -1 - return all possible tagged options
        """
        if n_tagged == -1:
            if self.lang == "en":
                # this to keep arpabet phonemes in the list of options
                if "[" in text and "]" in text:
                    tagged_texts = rewrite.rewrites(text, self.tagger.fst)
                else:
                    try:
                        tagged_texts = rewrite.rewrites(text, self.tagger.fst_no_digits)
                    except pynini.lib.rewrite.Error:
                        tagged_texts = rewrite.rewrites(text, self.tagger.fst)
            else:
                tagged_texts = rewrite.rewrites(text, self.tagger.fst)
        else:
            if self.lang == "en":
                # this to keep arpabet phonemes in the list of options
                if "[" in text and "]" in text:
                    tagged_texts = rewrite.top_rewrites(text, self.tagger.fst, nshortest=n_tagged)
                else:
                    try:
                        # try self.tagger graph that produces output without digits
                        tagged_texts = rewrite.top_rewrites(
                            text, self.tagger.fst_no_digits, nshortest=n_tagged
                        )
                    except pynini.lib.rewrite.Error:
                        tagged_texts = rewrite.top_rewrites(
                            text, self.tagger.fst, nshortest=n_tagged
                        )
            else:
                tagged_texts = rewrite.top_rewrites(text, self.tagger.fst, nshortest=n_tagged)
        return tagged_texts

    def _verbalize(self, tagged_text: str, normalized_texts: List[str], verbose: bool = False):
        """
        Verbalizes tagged text

        Args:
            tagged_text: text with tags
            normalized_texts: list of possible normalization options
            verbose: if true prints intermediate classification results
        """

        def get_verbalized_text(tagged_text):
            return rewrite.rewrites(tagged_text, self.verbalizer.fst)

        self.parser(tagged_text)
        tokens = self.parser.parse()
        tags_reordered = self.generate_permutations(tokens)
        for tagged_text_reordered in tags_reordered:
            try:
                tagged_text_reordered = pynini.escape(tagged_text_reordered)
                normalized_texts.extend(get_verbalized_text(tagged_text_reordered))
                if verbose:
                    print(tagged_text_reordered)

            except pynini.lib.rewrite.Error:
                continue

    def select_best_match(
        self,
        normalized_texts: List[str],
        input_text: str,
        pred_text: str,
        verbose: bool = False,
        remove_punct: bool = False,
        cer_threshold: int = 100,
    ):
        """
        Selects the best normalization option based on the lowest CER

        Args:
            normalized_texts: normalized text options
            input_text: input text
            pred_text: ASR model transcript of the audio file corresponding to the normalized text
            verbose: whether to print intermediate meta information
            remove_punct: whether to remove punctuation before calculating CER
            cer_threshold: if CER for pred_text is above the cer_threshold, no normalization will be performed

        Returns:
            normalized text with the lowest CER and CER value
        """
        if pred_text == "":
            return input_text, cer_threshold

        normalized_texts_cer = calculate_cer(normalized_texts, pred_text, remove_punct)
        normalized_texts_cer = sorted(normalized_texts_cer, key=lambda x: x[1])
        normalized_text, cer = normalized_texts_cer[0]

        if cer > cer_threshold:
            return input_text, cer

        if verbose:
            print("-" * 30)
            for option in normalized_texts:
                print(option)
            print("-" * 30)
        return normalized_text, cer


def calculate_cer(
    normalized_texts: List[str], pred_text: str, remove_punct=False
) -> List[Tuple[str, float]]:
    """
    Calculates character error rate (CER)

    Args:
        normalized_texts: normalized text options
        pred_text: ASR model output

    Returns: normalized options with corresponding CER
    """
    normalized_options = []
    for text in normalized_texts:
        text_clean = text.replace("-", " ").lower()
        if remove_punct:
            for punct in "!?:;,.-()*+-/<=>@^_":
                text_clean = text_clean.replace(punct, "")
        cer = round(word_error_rate([pred_text], [text_clean], use_cer=True) * 100, 2)
        normalized_options.append((text, cer))
    return normalized_options


def get_asr_model(asr_model):
    """
    Returns ASR Model

    Args:
        asr_model: NeMo ASR model
    """
    if os.path.exists(args.model):
        asr_model = ASRModel.restore_from(asr_model)
    elif args.model in ASRModel.get_available_model_names():
        asr_model = ASRModel.from_pretrained(asr_model)
    else:
        raise ValueError(
            f"Provide path to the pretrained checkpoint or choose from {ASRModel.get_available_model_names()}"
        )
    return asr_model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--text", help="input string or path to a .txt file", default=None, type=str
    )
    parser.add_argument(
        "--input_case",
        help="input capitalization",
        choices=["lower_cased", "cased"],
        default="cased",
        type=str,
    )
    parser.add_argument(
        "--language",
        help="Select target language",
        choices=["en", "ru", "de", "es"],
        default="en",
        type=str,
    )
    parser.add_argument(
        "--audio_data", default=None, help="path to an audio file or .json manifest"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="QuartzNet15x5Base-En",
        help="Pre-trained model name or path to model checkpoint",
    )
    parser.add_argument(
        "--n_tagged",
        type=int,
        default=30,
        help="number of tagged options to consider, -1 - return all possible tagged options",
    )
    parser.add_argument("--verbose", help="print info for debugging", action="store_true")
    parser.add_argument(
        "--no_remove_punct_for_cer",
        help="Set to True to NOT remove punctuation before calculating CER",
        action="store_true",
    )
    parser.add_argument(
        "--no_punct_post_process",
        help="set to True to disable punctuation post processing",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite_cache", help="set to True to re-create .far grammar files", action="store_true"
    )
    parser.add_argument(
        "--whitelist", help="path to a file with with whitelist", default=None, type=str
    )
    parser.add_argument(
        "--cache_dir",
        help="path to a dir with .far grammar file. Set to None to avoid using cache",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--n_jobs", default=-2, type=int, help="The maximum number of concurrently running jobs"
    )
    parser.add_argument(
        "--lm",
        action="store_true",
        help="Set to True for WFST+LM. Only available for English right now.",
    )
    parser.add_argument(
        "--cer_threshold",
        default=100,
        type=int,
        help="if CER for pred_text is above the cer_threshold, no normalization will be performed",
    )
    parser.add_argument(
        "--batch_size", default=200, type=int, help="Number of examples for each process"
    )
    return parser.parse_args()


def _normalize_line(
    normalizer: NormalizerWithAudio,
    n_tagged,
    verbose,
    line: str,
    remove_punct,
    punct_post_process,
    cer_threshold,
):
    line = json.loads(line)
    pred_text = line["pred_text"]

    normalized_texts = normalizer.normalize(
        text=line["text"],
        verbose=verbose,
        n_tagged=n_tagged,
        punct_post_process=punct_post_process,
    )

    normalized_texts = set(normalized_texts)
    normalized_text, cer = normalizer.select_best_match(
        normalized_texts=normalized_texts,
        input_text=line["text"],
        pred_text=pred_text,
        verbose=verbose,
        remove_punct=remove_punct,
        cer_threshold=cer_threshold,
    )
    line["nemo_normalized"] = normalized_text
    line["CER_nemo_normalized"] = cer
    return line


def normalize_manifest(
    normalizer,
    audio_data: str,
    n_jobs: int,
    n_tagged: int,
    remove_punct: bool,
    punct_post_process: bool,
    batch_size: int,
    cer_threshold: int,
):
    """
    Args:
        args.audio_data: path to .json manifest file.
    """

    def __process_batch(batch_idx: int, batch: List[str], dir_name: str):
        """
        Normalizes batch of text sequences
        Args:
            batch: list of texts
            batch_idx: batch index
            dir_name: path to output directory to save results
        """
        normalized_lines = [
            _normalize_line(
                normalizer,
                n_tagged,
                verbose=False,
                line=line,
                remove_punct=remove_punct,
                punct_post_process=punct_post_process,
                cer_threshold=cer_threshold,
            )
            for line in tqdm(batch)
        ]

        with open(f"{dir_name}/{batch_idx:05}.json", "w") as f_out:
            for line in normalized_lines:
                f_out.write(json.dumps(line, ensure_ascii=False) + "\n")

        print(f"Batch -- {batch_idx} -- is complete")

    manifest_out = audio_data.replace(".json", "_normalized.json")
    with open(audio_data, "r") as f:
        lines = f.readlines()

    print(f"Normalizing {len(lines)} lines of {audio_data}...")

    # to save intermediate results to a file
    batch = min(len(lines), batch_size)

    tmp_dir = manifest_out.replace(".json", "_parts")
    os.makedirs(tmp_dir, exist_ok=True)

    Parallel(n_jobs=n_jobs)(
        delayed(__process_batch)(idx, lines[i : i + batch], tmp_dir)
        for idx, i in enumerate(range(0, len(lines), batch))
    )

    # aggregate all intermediate files
    with open(manifest_out, "w") as f_out:
        for batch_f in sorted(glob(f"{tmp_dir}/*.json")):
            with open(batch_f, "r") as f_in:
                lines = f_in.read()
            f_out.write(lines)

    print(f"Normalized version saved at {manifest_out}")


if __name__ == "__main__":
    args = parse_args()

    if not ASR_AVAILABLE and args.audio_data:
        raise ValueError("NeMo ASR collection is not installed.")
    start = time.time()
    args.whitelist = os.path.abspath(args.whitelist) if args.whitelist else None
    if args.text is not None:
        normalizer = NormalizerWithAudio(
            input_case=args.input_case,
            lang=args.language,
            cache_dir=args.cache_dir,
            overwrite_cache=args.overwrite_cache,
            whitelist=args.whitelist,
            lm=args.lm,
        )

        if os.path.exists(args.text):
            with open(args.text, "r") as f:
                args.text = f.read().strip()
        normalized_texts = normalizer.normalize(
            text=args.text,
            verbose=args.verbose,
            n_tagged=args.n_tagged,
            punct_post_process=not args.no_punct_post_process,
        )

        if not normalizer.lm:
            normalized_texts = set(normalized_texts)
        if args.audio_data:
            asr_model = get_asr_model(args.model)
            pred_text = asr_model.transcribe([args.audio_data])[0]
            normalized_text, cer = normalizer.select_best_match(
                normalized_texts=normalized_texts,
                pred_text=pred_text,
                input_text=args.text,
                verbose=args.verbose,
                remove_punct=not args.no_remove_punct_for_cer,
                cer_threshold=args.cer_threshold,
            )
            print(f"Transcript: {pred_text}")
            print(f"Normalized: {normalized_text}")
        else:
            print("Normalization options:")
            for norm_text in normalized_texts:
                print(norm_text)
    elif not os.path.exists(args.audio_data):
        raise ValueError(f"{args.audio_data} not found.")
    elif args.audio_data.endswith(".json"):
        normalizer = NormalizerWithAudio(
            input_case=args.input_case,
            lang=args.language,
            cache_dir=args.cache_dir,
            overwrite_cache=args.overwrite_cache,
            whitelist=args.whitelist,
        )
        normalize_manifest(
            normalizer=normalizer,
            audio_data=args.audio_data,
            n_jobs=args.n_jobs,
            n_tagged=args.n_tagged,
            remove_punct=not args.no_remove_punct_for_cer,
            punct_post_process=not args.no_punct_post_process,
            batch_size=args.batch_size,
            cer_threshold=args.cer_threshold,
        )
    else:
        raise ValueError(
            "Provide either path to .json manifest in '--audio_data' OR "
            + "'--audio_data' path to audio file and '--text' path to a text file OR"
            "'--text' string text (for debugging without audio)"
        )
    print(f"Execution time: {round((time.time() - start)/60, 2)} min.")
