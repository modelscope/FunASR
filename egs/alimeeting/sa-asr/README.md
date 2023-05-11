# Get Started
Speaker Attributed Automatic Speech Recognition (SA-ASR) is a task proposed to solve "who spoke what". Specifically, the goal of SA-ASR is not only to obtain multi-speaker transcriptions, but also to identify the corresponding speaker for each utterance. The method used in this example is referenced in the paper: [End-to-End Speaker-Attributed ASR with Transformer](https://www.isca-speech.org/archive/pdfs/interspeech_2021/kanda21b_interspeech.pdf).  
To run this receipe, first you need to install FunASR and ModelScope. ([installation](https://alibaba-damo-academy.github.io/FunASR/en/installation.html))  
There are two startup scripts, `run.sh` for training and evaluating on the old eval and test sets, and `run_m2met_2023_infer.sh` for inference on the new test set of the Multi-Channel Multi-Party Meeting Transcription 2.0 ([M2MeT2.0](https://alibaba-damo-academy.github.io/FunASR/m2met2/index.html)) Challenge.  
Before running `run.sh`, you must manually download and unpack the [AliMeeting](http://www.openslr.org/119/) corpus and place it in the `./dataset` directory:
```shell
dataset
|—— Eval_Ali_far
|—— Eval_Ali_near
|—— Test_Ali_far
|—— Test_Ali_near
|—— Train_Ali_far
|—— Train_Ali_near
```
There are 16 stages in `run.sh`:
```shell
stage 1 - 5: Data preparation and processing.
stage 6: Generate speaker profiles (Stage 6 takes a lot of time).
stage 7 - 9: Language model training (Optional).
stage 10 - 11: ASR training (SA-ASR requires loading the pre-trained ASR model).
stage 12: SA-ASR training.
stage 13 - 16: Inference and evaluation.
```
Before running `run_m2met_2023_infer.sh`, you need to place the new test set `Test_2023_Ali_far` (to be released after the challenge starts) in the `./dataset` directory, which contains only raw audios. Then put the given `wav.scp`, `wav_raw.scp`, `segments`, `utt2spk` and `spk2utt` in the `./data/Test_2023_Ali_far` directory.  
```shell
data/Test_2023_Ali_far
|—— wav.scp
|—— wav_raw.scp
|—— segments
|—— utt2spk
|—— spk2utt
```
There are 4 stages in `run_m2met_2023_infer.sh`:
```shell
stage 1: Data preparation and processing.
stage 2: Generate speaker profiles for inference.
stage 3: Inference.
stage 4: Generation of SA-ASR results required for final submission.
```

The baseline model is available on [ModelScope](https://www.modelscope.cn/models/damo/speech_saasr_asr-zh-cn-16k-alimeeting/summary).
After generate stats of AliMeeting corpus(stage 10 in `run.sh`), you can set the `infer_with_pretrained_model=true` in `run.sh` to infer with our official baseline model released on ModelScope without training.

# Format of Final Submission
Finally, you need to submit a file called `text_spk_merge` with the following format:
```shell
Meeting_1 text_spk_1_A$text_spk_1_B$text_spk_1_C ...
Meeting_2 text_spk_2_A$text_spk_2_B$text_spk_2_C ...
...
```
Here, text_spk_1_A represents the full transcription of speaker_A of Meeting_1 (merged in chronological order), and $ represents the separator symbol. There's no need to worry about the speaker permutation as the optimal permutation will be computed in the end.  For more information, please refer to the results generated after executing the baseline code.
# Baseline Results
The results of the baseline system are as follows. The baseline results include speaker independent character error rate (SI-CER) and concatenated minimum permutation character error rate (cpCER), the former is speaker independent and the latter is speaker dependent. The speaker profile adopts the oracle speaker embedding during training. However, due to the lack of oracle speaker label during evaluation, the speaker profile provided by an additional spectral clustering is used. Meanwhile, the results of using the oracle speaker profile on Eval and Test Set are also provided to show the impact of speaker profile accuracy.  
<table>
    <tr >
	    <td rowspan="2"></td>
        <td colspan="2">SI-CER(%)</td>
	    <td colspan="2">cpCER(%)</td>
	</tr>
    <tr>
        <td>Eval</td>
	    <td>Test</td>
	    <td>Eval</td>
	    <td>Test</td>
	</tr>
    <tr>
	    <td>oracle profile</td>
        <td>32.05</td>
        <td>32.70</td>
	    <td>47.40</td>
        <td>52.57</td>
	</tr>
    <tr>
	    <td>cluster profile</td>
        <td>32.05</td>
        <td>32.70</td>
	    <td>53.76</td>
        <td>55.95</td>
	</tr>
</table>

# Reference
N. Kanda, G. Ye, Y. Gaur, X. Wang, Z. Meng, Z. Chen, and T. Yoshioka, "End-to-end speaker-attributed ASR with transformer," in Interspeech. ISCA, 2021, pp. 4413–4417.