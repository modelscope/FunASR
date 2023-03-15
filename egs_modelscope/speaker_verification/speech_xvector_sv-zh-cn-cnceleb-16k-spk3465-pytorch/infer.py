from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np

if __name__ == '__main__':
    inference_sv_pipline = pipeline(
        task=Tasks.speaker_verification,
        model='damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch'
    )

    # 提取不同句子的说话人嵌入码
    # for url use "utt_id" as key
    rec_result = inference_sv_pipline(
        audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_enroll.wav')
    enroll = rec_result["spk_embedding"]

    # for local file use "utt_id" as key
    rec_result = inference_sv_pipline(audio_in='sv_example_same.wav')["test1"]
    same = rec_result["spk_embedding"]

    import soundfile
    wav = soundfile.read('sv_example_enroll.wav')[0]
    # for raw inputs use "utt_id" as key
    spk_embedding = inference_sv_pipline(audio_in=wav)["spk_embedding"]

    rec_result = inference_sv_pipline(
        audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_different.wav')
    different = rec_result["spk_embedding"]

    # 对相同的说话人计算余弦相似度
    sv_threshold = 0.9465
    same_cos = np.sum(enroll * same) / (np.linalg.norm(enroll) * np.linalg.norm(same))
    same_cos = max(same_cos - sv_threshold, 0.0) / (1.0 - sv_threshold) * 100.0
    print("Similarity:", same_cos)

    # 对不同的说话人计算余弦相似度
    diff_cos = np.sum(enroll * different) / (np.linalg.norm(enroll) * np.linalg.norm(different))
    diff_cos = max(diff_cos - sv_threshold, 0.0) / (1.0 - sv_threshold) * 100.0
    print("Similarity:", diff_cos)
