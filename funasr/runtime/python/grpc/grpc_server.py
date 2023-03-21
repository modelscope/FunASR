from concurrent import futures
import grpc
import json
import time

import paraformer_pb2_grpc
from paraformer_pb2 import Response


class ASRServicer(paraformer_pb2_grpc.ASRServicer):
    def __init__(self, user_allowed, model, sample_rate, backend, onnx_dir, vad_model='', punc_model=''):
        print("ASRServicer init")
        self.backend = backend
        self.init_flag = 0
        self.client_buffers = {}
        self.client_transcription = {}
        self.auth_user = user_allowed.split("|")
        if self.backend == "pipeline":
            try:
                from modelscope.pipelines import pipeline
                from modelscope.utils.constant import Tasks
            except ImportError:
                raise ImportError(f"Please install modelscope")
            self.inference_16k_pipeline = pipeline(task=Tasks.auto_speech_recognition, model=model, vad_model=vad_model, punc_model=punc_model)
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
            elif req.isEnd: #end grpc
                print("asr end")
                self.disconnect(req.user)
                result = {}
                result["success"] = True
                result["detail"] = "asr end"
                result["text"] = ""
                yield Response(sentence=json.dumps(result), user=req.user, action="terminate",language=req.language)
            elif req.speaking: #continue speaking
                if req.audio_data is not None and len(req.audio_data) > 0:
                    if req.user in self.client_buffers:
                        self.client_buffers[req.user] += req.audio_data #append audio
                    else:
                        self.client_buffers[req.user] = req.audio_data
                result = {}
                result["success"] = True
                result["detail"] = "speaking"
                result["text"] = ""
                yield Response(sentence=json.dumps(result), user=req.user, action="speaking", language=req.language)
            elif not req.speaking: #silence
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
                    if len(tmp_data) < 9600: #min input_len for asr model , 300ms
                        end_time = int(round(time.time() * 1000))
                        delay_str = str(end_time - begin_time)
                        result = {}
                        result["success"] = True
                        result["detail"] = "waiting_for_more_voice"
                        result["server_delay_ms"] = delay_str
                        result["text"] = ""
                        print ("user: %s , delay(ms): %s, info: %s " % (req.user, delay_str, "waiting_for_more_voice"))
                        yield Response(sentence=json.dumps(result), user=req.user, action="waiting", language=req.language)
                    else:
                        if self.backend == "pipeline":
                            asr_result = self.inference_16k_pipeline(audio_in=tmp_data, audio_fs = self.sample_rate)
                            if "text" in asr_result:
                                asr_result = asr_result['text']
                            else:
                                asr_result = ""
                        elif self.backend == "onnxruntime":
                            from rapid_paraformer.utils.frontend import load_bytes
                            array = load_bytes(tmp_data)
                            asr_result = self.inference_16k_pipeline(array)[0]
                        end_time = int(round(time.time() * 1000))
                        delay_str = str(end_time - begin_time)
                        print ("user: %s , delay(ms): %s, text: %s " % (req.user, delay_str, asr_result))
                        result = {}
                        result["success"] = True
                        result["detail"] = "finish_sentence"
                        result["server_delay_ms"] = delay_str
                        result["text"] = asr_result
                        yield Response(sentence=json.dumps(result), user=req.user, action="finish", language=req.language)
            else:
                result = {}
                result["success"] = False 
                result["detail"] = "error, no condition matched! Unknown reason."
                result["text"] = ""
                self.disconnect(req.user)
                yield Response(sentence=json.dumps(result), user=req.user, action="terminate", language=req.language)
                

