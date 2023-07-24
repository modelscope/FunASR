# Get Started
Speaker Attributed Automatic Speech Recognition (SA-ASR) is a task proposed to solve "who spoke what". Specifically, the goal of SA-ASR is not only to obtain multi-speaker transcriptions, but also to identify the corresponding speaker for each utterance. The method used in this example is referenced in the paper: [End-to-End Speaker-Attributed ASR with Transformer](https://www.isca-speech.org/archive/pdfs/interspeech_2021/kanda21b_interspeech.pdf).  
# Train
First you need to install the FunASR and ModelScope. ([installation](https://github.com/alibaba-damo-academy/FunASR#installation))
After the FunASR and ModelScope is installed, you must manually download and unpack the [AliMeeting](http://www.openslr.org/119/) corpus and place it in the `./dataset` directory. The `.dataset` should organized as follow:
```shell
dataset
|—— Eval_Ali_far
|—— Eval_Ali_near
|—— Test_Ali_far
|—— Test_Ali_near
|—— Train_Ali_far
|—— Train_Ali_near
```
Then you can run this receipe by running:
```shell
bash run.sh --stage 0 --stop-stage 6
```
There are 8 stages in `run.sh`:
```shell
stage 0: Data preparation and remove the audio which is too long or too short.
stage 1: Speaker profile and CMVN Generation.
stage 2: Dictionary preparation.
stage 3: LM training (not supported).
stage 4: ASR Training.
stage 5: SA-ASR Training.
stage 6: Inference
stage 7: Inference with Test_2023_Ali_far
```
<!-- The baseline model is available on [ModelScope](https://www.modelscope.cn/models/damo/speech_saasr_asr-zh-cn-16k-alimeeting/summary). -->
# Infer
1. Download the final test set and extracted
2. Put the audios in `./dataset/Test_2023_Ali_far/` and put the `wav.scp`, `segments`, `utt2spk`, `spk2utt` in `./data/org/Test_2023_Ali_far/`.
3. Set the `test_2023` in `run.sh` should be  to `Test_2023_Ali_far`.
4. Run the `run.sh` as follow.
```shell
# Prepare test_2023 set
bash run.sh --stage 0 --stop-stage 1
# Decode test_2023 set
bash run.sh --stage 7 --stop-stage 7
```
# Format of Final Submission
Finally, you need to submit a file called `text_spk_merge` with the following format:
```shell
Meeting_1 text_spk_1_A$text_spk_1_B$text_spk_1_C ...
Meeting_2 text_spk_2_A$text_spk_2_B$text_spk_2_C ...
...
```
Here, text_spk_1_A represents the full transcription of speaker_A of Meeting_1 (merged in chronological order), and $ represents the separator symbol. There's no need to worry about the speaker permutation as the optimal permutation will be computed in the end.  For more information, please refer to the results generated after executing the baseline code.
# Baseline Results
The results of the baseline system are as follows. The baseline results include speaker independent character error rate (SI-CER) and concatenated minimum permutation character error rate (cpCER), the former is speaker independent and the latter is speaker dependent. The speaker profile adopts the oracle speaker embedding during training. However, due to the lack of oracle speaker label during evaluation, the speaker profile provided by an additional spectral clustering is used. Meanwhile, the results of using the oracle speaker profile on Test Set are also provided to show the impact of speaker profile accuracy.  
<!-- <table>
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
        <td>32.72</td>
	    <td>47.40</td>
        <td>42.92</td>
	</tr>
    <tr>
	    <td>cluster profile</td>
        <td>32.05</td>
        <td>32.73</td>
	    <td>53.76</td>
        <td>49.37</td>
	</tr>
</table> -->
|                |SI-CER(%)     |cpCER(%)  |
|:---------------|:------------:|----------:|
|oracle profile  |32.72         |42.92      |
|cluster  profile|32.73         |49.37      |


# Reference
N. Kanda, G. Ye, Y. Gaur, X. Wang, Z. Meng, Z. Chen, and T. Yoshioka, "End-to-end speaker-attributed ASR with transformer," in Interspeech. ISCA, 2021, pp. 4413–4417.