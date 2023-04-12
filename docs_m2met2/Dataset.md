# Datasets
## Overview of training data
In the fixed training condition, the training dataset is restricted to three publicly available corpora, namely, AliMeeting, AISHELL-4, and CN-Celeb. To evaluate the performance of the models trained on these datasets, we will release a new Test set called Test-2023 for scoring and ranking. We will describe the AliMeeting dataset and the Test-2023 set in detail.
## Detail of AliMeeting corpus
AliMeeting contains 118.75 hours of speech data in total. The dataset is divided into 104.75 hours for training (Train), 4 hours for evaluation (Eval) and 10 hours as test set (Test) for scoring and ranking. Specifically, the Train and Eval sets contain 212 and 8 sessions, respectively. Each session consists of a 15 to 30-minute discussion by a group of participants. The total number of participants in Train and Eval sets is 456 and 25, respectively, with balanced gender coverage.

The dataset is collected in 13 meeting venues, which are categorized into three types: small, medium, and large rooms with sizes ranging from 8 m$^{2}$ to 55 m$^{2}$. Different rooms give us a variety of acoustic properties and layouts. The detailed parameters of each meeting venue will be released together with the Train data. The type of wall material of the meeting venues covers cement, glass, etc. Other furnishings in meeting venues include sofa, TV, blackboard, fan, air conditioner, plants, etc. During recording, the participants of the meeting sit around the microphone array which is placed on the table and conduct a natural conversation. The microphone-speaker distance ranges from 0.3 m to 5.0 m. All participants are native Chinese speakers speaking Mandarin without strong accents. During the meeting, various kinds of indoor noise including but not limited to clicking, keyboard, door opening/closing, fan, bubble noise, etc., are made naturally. For both Train and Eval sets, the participants are required to remain in the same position during recording. There is no speaker overlap between the Train and Eval set. An example of the recording venue from the Train set is shown in Fig 1.

![meeting room](images/meeting_room.png)

The number of participants within one meeting session ranges from 2 to 4. To ensure the coverage of different overlap ratios, we select various meeting topics during recording, including medical treatment, education, business, organization management, industrial production and other daily routine meetings. The average speech overlap ratio of Train and Eval sets are 42.27\% and 34.76\%, respectively. More details of AliMeeting are shown in Table 1. A detailed overlap ratio distribution of meeting sessions with different numbers of speakers in the Train and Eval set is shown in Table 2.

![dataset detail](images/dataset_detail.png)

The Test-2023 set consists of 20 sessions that were recorded in an identical acoustic setting to that of the AliMeeting corpus. Each meeting session in the Test-2023 dataset comprises between 2 and 4 participants, thereby sharing a similar configuration with the AliMeeting test set.

We also record the near-field signal of each participant using a headset microphone and ensure that only the participant's own speech is recorded and transcribed. It is worth noting that the far-field audio recorded by the microphone array and the near-field audio recorded by the headset microphone will be synchronized to a common timeline range.

All transcriptions of the speech data are prepared in TextGrid format for each session, which contains the information of the session duration, speaker information (number of speaker, speaker-id, gender, etc.), the total number of segments of each speaker, the timestamp and transcription of each segment, etc.
## Get the data
The three dataset for training mentioned above can be downloaded at [OpenSLR](https://openslr.org/resources.php). The participants can download via the following links. Particularly, in the baseline we provide convenient data preparation scripts for AliMeeting corpus.
- [AliMeeting](https://openslr.org/119/)
- [AISHELL-4](https://openslr.org/111/)
- [CN-Celeb](https://openslr.org/82/)