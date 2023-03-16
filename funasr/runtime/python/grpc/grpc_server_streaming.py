from concurrent import futures
import grpc
import json
import time
import numpy as np

import paraformer_pb2_grpc
from paraformer_pb2 import Response


class VAD_ASR_PUNC_Servicer(paraformer_pb2_grpc.ASRServicer):
    def __init__(self, user_allowed, vad_model, asr_model, punc_model, sample_rate, backend, onnx_dir):
        print("ASRServicer init")
        self.backend = backend
        self.init_flag = 0
        self.client_buffers = {}
        self.client_transcription = {}
        self.auth_user = user_allowed.split("|")

        self.result_buffer = ""
        self.audio_buffers = None
        self.asr_buffers = None
        self.asr_cache = {"encoder": {"start_idx": 0, "pad_left": 5, "stride": 10, "pad_right": 5, "cif_hidden": None, "cif_alphas": None}, "decoder": {"decode_fsmn": None}}
        self.vad_cache = {"in_cache": dict(), "is_final": False}
        self.punc_cache = []

        self.speech_start_index = 0
        self.speech_start = False
        self.speech_end = False
        self.first_chunk = True
        self.chunk_index = 1
        if self.backend == "pipeline":
            try:
                from modelscope.pipelines import pipeline
                from modelscope.utils.constant import Tasks
            except ImportError:
                raise ImportError(f"Please install modelscope")
            self.asr_inference_pipeline = pipeline(task=Tasks.auto_speech_recognition, model=asr_model, model_revision='v1.0.0')
            self.vad_inference_pipeline = pipeline(task=Tasks.voice_activity_detection, model=vad_model, model_revision='v1.1.9')
            self.punc_inference_pipeline = pipeline(task=Tasks.punctuation, model=punc_model, model_revision="v1.0.1")

        elif self.backend == "onnxruntime":
            try:
                from rapid_paraformer.paraformer_onnx import Paraformer
            except ImportError:
                raise ImportError(f"Please install onnxruntime environment")
            self.inference_16k_pipeline = Paraformer(model_dir=onnx_dir)
        self.sample_rate = sample_rate

    def clear_states(self, user):
        self.clear_buffers(user)
        self.clear_transcriptions(user)

    def clear_buffers(self, user):
        if user in self.client_buffers:
            del self.client_buffers[user]

    def clear_transcriptions(self, user):
        if user in self.client_transcription:
            del self.client_transcription[user]

    def disconnect(self, user):
        self.clear_states(user)
        print("Disconnecting user: %s" % str(user))

    def Recognize(self, request_iterator, context):

        for req in request_iterator:
            if req.user not in self.auth_user:
                result = {}
                result["success"] = False
                result["detail"] = "Not Authorized user: %s " % req.user
                result["text"] = ""
                yield Response(sentence=json.dumps(result), user=req.user, action="terminate", language=req.language)
            elif req.isEnd:  # end grpc
                print("asr end")
                self.disconnect(req.user)
                result = {}
                result["success"] = True
                result["detail"] = "asr end"
                result["text"] = ""
                yield Response(sentence=json.dumps(result), user=req.user, action="terminate", language=req.language)
            elif req.speaking:  # continue speaking
                if req.audio_data is not None and len(req.audio_data) > 0:
                    if req.user in self.client_buffers:
                        self.client_buffers[req.user] += req.audio_data  # append audio
                    else:
                        self.client_buffers[req.user] = req.audio_data
                result = {}
                result["success"] = True
                result["detail"] = "speaking"
                result["text"] = ""
                yield Response(sentence=json.dumps(result), user=req.user, action="speaking", language=req.language)
            elif not req.speaking:  # silence
                if req.user not in self.client_buffers:
                    result = {}
                    result["success"] = True
                    result["detail"] = "waiting_for_more_voice"
                    result["text"] = ""
                    yield Response(sentence=json.dumps(result), user=req.user, action="waiting", language=req.language)
                else:
                    begin_time = int(round(time.time() * 1000))
                    tmp_data = self.client_buffers[req.user]
                    self.clear_states(req.user)
                    result = {}
                    result["success"] = True
                    result["detail"] = "decoding data: %d bytes" % len(tmp_data)
                    result["text"] = ""
                    yield Response(sentence=json.dumps(result), user=req.user, action="decoding", language=req.language)
                    if len(tmp_data) < 800:  # min input_len for asr model , 300ms
                        end_time = int(round(time.time() * 1000))
                        delay_str = str(end_time - begin_time)
                        result = {}
                        result["success"] = True
                        result["detail"] = "waiting_for_more_voice"
                        result["server_delay_ms"] = delay_str
                        result["text"] = ""
                        print("user: %s , delay(ms): %s, info: %s " % (req.user, delay_str, "waiting_for_more_voice"))
                        yield Response(sentence=json.dumps(result), user=req.user, action="waiting",
                                       language=req.language)
                    else:
                        from rapid_paraformer.utils.frontend import load_bytes
                        tmp_data = load_bytes(tmp_data)
                        if self.backend == "pipeline":
                            if self.audio_buffers is None:
                                self.audio_buffers = tmp_data
                            else:
                                self.audio_buffers = np.append(self.audio_buffers, tmp_data)
                            vad_result = self.vad_inference_pipeline(audio_in=tmp_data, param_dict=self.vad_cache)
                            print(vad_result)
                            if len(vad_result) == 0:
                                #pass
                                if self.speech_start:
                                    if self.asr_buffers is None:
                                        self.asr_buffers = self.audio_buffers[self.speech_start_index:-12800]
                                    else:
                                        self.asr_buffers = np.append(self.asr_buffers, self.audio_buffers[self.speech_start_index:-12800])
                                self.speech_start_index = max(self.speech_start_index, len(self.audio_buffers) - 12800)
                            else:
                                vad_segment = json.loads(vad_result["text"])
                                for i in range(len(vad_segment)):
                                    if vad_segment[i][0] != -1:
                                        self.speech_start_index = vad_segment[i][0] * (self.sample_rate // 1000)
                                        self.speech_start = True
                                        self.speech_end = False
                                    elif vad_segment[i][1] != -1:
                                        if self.asr_buffers is None:
                                            self.asr_buffers = self.audio_buffers[self.speech_start_index:vad_segment[i][1] * (self.sample_rate // 1000)]
                                        else:
                                            self.asr_buffers = np.append(self.asr_buffers, self.audio_buffers[self.speech_start_index:vad_segment[i][1] * (self.sample_rate // 1000)])
                                        self.speech_start = False  
                                        self.speech_end = True
                            if self.audio_buffers is not None:
                                print(len(self.audio_buffers))
                            if self.audio_buffers is not None:
                                if self.first_chunk:
                                    if len(self.audio_buffers) >= 14400:
                                        self.asr_cache["encoder"]["pad_left"] = 0
                                        asr_result = self.asr_inference_pipeline(audio_in=self.audio_buffers[0:14400], param_dict={"cache":self.asr_cache})
                                        print("-------asr result {}".format(asr_result))
                                        asr_result = asr_result["text"]
                                        if asr_result != "sil":
                                            self.result_buffer += asr_result
                                        self.first_chunk = False
                                        self.chunk_index += 1

                                        if self.speech_end:
                                            if self.result_buffer == "sil":
                                                pass
                                            else:
                                                pass
                                                #punc_result = self.punc_inference_pipeline(text_in=asr_result,
                                                #                                           cache=self.punc_cache)
                                                #self.punc_cache = punc_result["cache"]
                                                #asr_result = punc_result["text"]
                                                #self.result_buffer = ""

                                        end_time = int(round(time.time() * 1000))
                                        delay_str = str(end_time - begin_time)
                                        print("user: %s , delay(ms): %s, text: %s " % (req.user, delay_str, asr_result))
                                        result = {"success": True, "detail": "finish_sentence",
                                                  "server_delay_ms": delay_str, "text": asr_result}
                                        yield Response(sentence=json.dumps(result), user=req.user, action="finish",
                                                       language=req.language)

                                else:
                                    while len(self.audio_buffers) >= 14400 + (self.chunk_index - 1)*9600:
                                        self.asr_cache["encoder"]["pad_left"] = 5
                                        start_idx = 5 + (self.chunk_index - 2) * 10
                                        self.asr_cache["encoder"]["start_idx"] = start_idx
                                        print(start_idx)
                                        print(self.chunk_index)
                                        asr_result = self.asr_inference_pipeline(audio_in=self.audio_buffers[start_idx*960:start_idx*960+19200],
                                                                                 param_dict={"cache":self.asr_cache})
                                        print("-------asr result {}".format(asr_result))
                                        self.chunk_index += 1
                                        asr_result = asr_result["text"]
                                        if asr_result != "sil":
                                            self.result_buffer += asr_result

                                        if self.speech_end:
                                            if self.result_buffer == "sil":
                                                pass
                                            else:
                                                pass
                                                #punc_result = self.punc_inference_pipeline(text_in=asr_result,
                                                #                                           cache=self.punc_cache)
                                                #self.punc_cache = punc_result["cache"]
                                                #asr_result = punc_result["text"]
                                                #self.result_buffer = ""

                                        end_time = int(round(time.time() * 1000))
                                        delay_str = str(end_time - begin_time)
                                        print("user: %s , delay(ms): %s, text: %s " % (req.user, delay_str, asr_result))
                                        result = {"success": True, "detail": "finish_sentence",
                                                  "server_delay_ms": delay_str, "text": asr_result}
                                        yield Response(sentence=json.dumps(result), user=req.user, action="finish",
                                                       language=req.language)

                        elif self.backend == "onnxruntime":
                            from rapid_paraformer.utils.frontend import load_bytes
                            array = load_bytes(tmp_data)
                            asr_result = self.inference_16k_pipeline(array)[0]
                            end_time = int(round(time.time() * 1000))
                            delay_str = str(end_time - begin_time)
                            print("user: %s , delay(ms): %s, text: %s " % (req.user, delay_str, asr_result))
                            result = {"success": True, "detail": "finish_sentence", "server_delay_ms": delay_str,
                                      "text": asr_result}
                            yield Response(sentence=json.dumps(result), user=req.user, action="finish",
                                           language=req.language)

                        end_time = int(round(time.time() * 1000))
                        delay_str = str(end_time - begin_time)
                        result = {"success": True, "detail": "waiting_for_more_voice", "server_delay_ms": delay_str,
                                  "text": ""}
                        print("user: %s , delay(ms): %s, info: %s " % (req.user, delay_str, "waiting_for_more_voice"))
                        yield Response(sentence=json.dumps(result), user=req.user, action="waiting",
                                       language=req.language)
            else:
                result = {}
                result["success"] = False
                result["detail"] = "error, no condition matched! Unknown reason."
                result["text"] = ""
                self.disconnect(req.user)
                yield Response(sentence=json.dumps(result), user=req.user, action="terminate", language=req.language)

