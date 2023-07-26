# Introduction
## Call for participation
Automatic speech recognition (ASR) and speaker diarization have made significant strides in recent years, resulting in a surge of speech technology applications across various domains. However, meetings present unique challenges to speech technologies due to their complex acoustic conditions and diverse speaking styles, including overlapping speech, variable numbers of speakers, far-field signals in large conference rooms, and environmental noise and reverberation. 

Over the years, several challenges have been organized to advance the development of meeting transcription, including the Rich Transcription evaluation and Computational Hearing in Multisource Environments (CHIME) challenges. The latest iteration of the CHIME challenge has a particular focus on distant automatic speech recognition and developing systems that can generalize across various array topologies and application scenarios. However, while progress has been made in English meeting transcription, language differences remain a significant barrier to achieving comparable results in non-English languages, such as Mandarin. The Multimodal Information Based Speech Processing (MISP) and Multi-Channel Multi-Party Meeting Transcription (M2MeT) challenges have been instrumental in advancing Mandarin meeting transcription. The MISP challenge seeks to address the problem of audio-visual distant multi-microphone signal processing in everyday home environments, while the M2MeT challenge focuses on tackling the speech overlap issue in offline meeting rooms.

The ICASSP2022 M2MeT challenge focuses on meeting scenarios, and it comprises two main tasks: speaker diarization and multi-speaker automatic speech recognition. The former involves identifying who spoke when in the meeting, while the latter aims to transcribe speech from multiple speakers simultaneously, which poses significant technical difficulties due to overlapping speech and acoustic interferences.

Building on the success of the previous M2MeT challenge, we are excited to propose the M2MeT2.0 challenge as an ASRU 2023 challenge special session. In the original M2MeT challenge, the evaluation metric was speaker-independent, which meant that the transcription could be determined, but not the corresponding speaker. To address this limitation and further advance the current multi-talker ASR system towards practicality, the M2MeT2.0 challenge proposes the speaker-attributed ASR task with two sub-tracks: fixed and open training conditions. The speaker-attribute automatic speech recognition (ASR) task aims to tackle the practical and challenging problem of identifying "who spoke what at when". To facilitate reproducible research in this field, we offer a comprehensive overview of the dataset, rules, evaluation metrics, and baseline systems. Furthermore, we will release a carefully curated test set, comprising approximately 10 hours of audio, according to the timeline. The new test set is designed to enable researchers to validate and compare their models' performance and advance the state of the art in this area.

## Timeline(AOE Time)
- $ April~29, 2023: $ Challenge and registration open.
- $ May~11, 2023: $ Baseline release.
- $ May~22, 2023: $ Registration deadline, the due date for participants to join the Challenge.
- $ June~16, 2023: $ Test data release and leaderboard open.
- $ June~20, 2023: $ Final submission deadline and leaderboar close.
- $ June~26, 2023: $ Evaluation result and ranking release.
- $ July~3, 2023: $ Deadline for paper submission.
- $ July~10, 2023: $ Deadline for final paper submission.
- $ December~12\ to\ 16, 2023: $ ASRU Workshop and Challenge Session.

## Guidelines

Interested participants, whether from academia or industry, must register for the challenge by completing the Google form below. The deadline for registration is May 22, 2023. 

[M2MeT2.0 Registration](https://docs.google.com/forms/d/e/1FAIpQLSf77T9vAl7Ym-u5g8gXu18SBofoWRaFShBo26Ym0-HDxHW9PQ/viewform?usp=sf_link)

Within three working days, the challenge organizer will send email invitations to eligible teams to participate in the challenge. All qualified teams are required to adhere to the challenge rules, which will be published on the challenge page. Prior to the ranking release time, each participant must submit a system description document detailing their approach and methods. The organizer will select the top ranking submissions to be included in the ASRU2023 Proceedings. 
